from dataclasses import dataclass
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


try:
    import flower_state_classification.cv.training.transforms as T
    import flower_state_classification.cv.models.trained_models as trained_models
    from flower_state_classification.cv.training.dataset_downloaders import get_coco_dataset
    from flower_state_classification.cv.training.pytorch_datasets import FiftyOneTorchDataset, add_detections
    from flower_state_classification.cv.training.engine import train_one_epoch, evaluate
    import flower_state_classification.cv.training.utils as utils
    model_folder = trained_models.__path__[0]

# imports when running with different python interpreter for model training
except:
    from dataset_downloaders import get_coco_dataset
    from pytorch_datasets import FiftyOneTorchDataset, add_detections
    from engine import train_one_epoch, evaluate
    import transforms as T
    import utils
    model_folder = "C:\\dev"

from fiftyone import ViewField as F
import fiftyone as fo

#get folder from imported package


@dataclass
class training_metadata:
    model_type: str
    model_name: str

    dataset_name: str
    dataset_comment: str

    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str

def create_pytorch_datasets(dataset, split=0.8, max_samples=None):
    train_transforms = T.Compose([T.PILToTensor(), T.ConvertImageDtype(torch.float32), T.RandomHorizontalFlip(0.5)])
    test_transforms = T.Compose([T.PILToTensor(), T.ConvertImageDtype(torch.float32)])
    # split the dataset in train and test set
    if max_samples:
        dataset = dataset.take(max_samples, seed=51)
    len_dataset = len(dataset)
    train_view = dataset.take(int(len_dataset*split), seed=51)
    test_view = dataset.exclude([s.id for s in train_view])
    # use our dataset and defined transformations
    torch_dataset = FiftyOneTorchDataset(train_view, train_transforms,
            classes=["potted plant"])
    torch_dataset_test = FiftyOneTorchDataset(test_view, test_transforms, 
            classes=["potted plant"])
    
    return torch_dataset, torch_dataset_test, train_view, test_view


def get_model(num_classes):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model



def do_training(model, torch_dataset, torch_dataset_test, num_epochs=4):
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        torch_dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)
    
    data_loader_test = torch.utils.data.DataLoader(
        torch_dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device %s" % device)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=3,
                                                    gamma=0.1)

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

 
def main():
    dataset = get_coco_dataset()
    dataset.compute_metadata()
    #session = fo.launch_app(dataset)
    #session.wait()
    dataset_view = dataset.match(F("ground_truth.detections").length() > 0)

    train_dataset, test_dataset, train_view, test_view = create_pytorch_datasets(dataset_view, max_samples=1000)
    pretrained_model = get_model(1+1) # include background for retinanet
    do_training(pretrained_model, train_dataset, test_dataset, num_epochs=1)
    add_detections(pretrained_model, test_dataset, dataset, field_name="predictions")
    torch.save(pretrained_model.state_dict(), model_folder + "/faster_rcnn_resnet50_fpn.pth")
    results = fo.evaluate_detections(
        test_view, 
        "predictions", 
        classes=["potted plant"], 
        eval_key="eval", 
        compute_mAP=True
    )
    results.mAP()
    results.print_report()
    session = fo.launch_app(dataset)
    session.view = test_view.sort_by("eval_fp", reverse=False)
    session.wait()

if __name__ == "__main__":
    main()