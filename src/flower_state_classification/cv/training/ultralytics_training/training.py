import numpy as np
import os
from tqdm import tqdm
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import fiftyone.utils.random as four

from ultralytics import YOLO

def read_yolo_detections_file(filepath):
    folder = r"E:\dev\flower-state-classification\src\flower_state_classification\cv\training\yolov8"
    filepath=os.path.join(folder,filepath)
    detections = []
    if not os.path.exists(filepath):
        return np.array([])
    
    with open(filepath) as f:
        lines = [line.rstrip('\n').split(' ') for line in f]
    
    for line in lines:
        detection = [float(l) for l in line]
        detections.append(detection)
    return np.array(detections)

def _uncenter_boxes(boxes):
    '''convert from center coords to corner coords'''
    boxes[:, 0] -= boxes[:, 2]/2.
    boxes[:, 1] -= boxes[:, 3]/2.

def _get_class_labels(predicted_classes, class_list):
    labels = (predicted_classes).astype(int)
    labels = [class_list[l] for l in labels]
    return labels

def convert_yolo_detections_to_fiftyone(
    yolo_detections, 
    class_list
    ):
    detections = []
    if yolo_detections.size == 0:
        return fo.Detections(detections=detections)
    
    boxes = yolo_detections[:, 1:-1]
    _uncenter_boxes(boxes)
    
    confs = yolo_detections[:, -1]
    labels = _get_class_labels(yolo_detections[:, 0], class_list)  
    for label, conf, box in zip(labels, confs, boxes):
        detections.append(
            fo.Detection(
                label=label,
                bounding_box=box.tolist(),
                confidence=conf
            )
        )
    return fo.Detections(detections=detections)

def convert_yolo_segmentations_to_fiftyone(
    yolo_segmentations, 
    class_list
    ):
    detections = []
    boxes = yolo_segmentations.boxes.xywhn
    if not boxes.shape or yolo_segmentations.masks is None:
        return fo.Detections(detections=detections)
    
    _uncenter_boxes(boxes)
    masks = yolo_segmentations.masks.masks
    labels = _get_class_labels(yolo_segmentations.boxes.cls, class_list)
    for label, box, mask in zip(labels, boxes, masks):
        ## convert to absolute indices to index mask
        w, h = mask.shape
        tmp =  np.copy(box)
        tmp[2] += tmp[0]
        tmp[3] += tmp[1]
        tmp[0] *= h
        tmp[2] *= h
        tmp[1] *= w
        tmp[3] *= w
        tmp = [int(b) for b in tmp]
        y0, x0, y1, x1 = tmp
        sub_mask = mask[x0:x1, y0:y1]
       
        detections.append(
            fo.Detection(
                label=label,
                bounding_box = list(box),
                mask = sub_mask.astype(bool)
            )
        )
    return fo.Detections(detections=detections)

def export_yolo_data(
    samples, 
    export_dir, 
    classes, 
    label_field = "ground_truth", 
    split = None
    ):
    if type(split) == list:
        splits = split
        for split in splits:
            export_yolo_data(
                samples, 
                export_dir, 
                classes, 
                label_field, 
                split
            )   
    else:
        if split is None:
            split_view = samples
            split = "val"
        else:
            split_view = samples.match_tags(split)
        split_view.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=label_field,
            classes=classes,
            split=split
        )

def get_prediction_filepath(filepath, run_number = 1):
    filepath = filepath.replace("\\", "/")
    run_num_string = ""
    if run_number != 1:
        run_num_string = str(run_number)
    filename = filepath.split("/")[-1].split(".")[0]
    return "runs/detect/predict{}/labels/".format(run_num_string) + filename + ".txt"

def add_yolo_detections(
    samples,
    prediction_field,
    prediction_filepath,
    class_list
    ):
    prediction_filepaths = samples.values(prediction_filepath)
    yolo_detections = [read_yolo_detections_file(pf) for pf in prediction_filepaths]
    detections =  [convert_yolo_detections_to_fiftyone(yd, class_list) for yd in yolo_detections]
    samples.set_values(prediction_field, detections)

def add_detection(dataset):
    detection_model = YOLO("yolov8n.pt")
    seg_model = YOLO("yolov8n-seg.pt")

    
    coco_classes = [c for c in dataset.default_classes if not c.isnumeric()]

    coco_val_dir = "coco_val"
    export_yolo_data(dataset, coco_val_dir, coco_classes)
    filepaths = dataset.values("filepath")
    prediction_filepaths = [get_prediction_filepath(fp) for fp in filepaths]
    dataset.set_values(
        "yolov8n_det_filepath", 
        prediction_filepaths
    )
    add_yolo_detections(
        dataset, 
        "yolov8n", 
        "yolov8n_det_filepath", 
        coco_classes
    )
    return dataset

def evaluate_dataset(dataset):
    detection_results = dataset.evaluate_detections(
        "yolov8n", 
        eval_key="eval",
        compute_mAP=True,
        gt_field="ground_truth",
    )
    mAP = detection_results.mAP()
    print("mAP = {}".format(mAP))
    counts = dataset.count_values("ground_truth.detections.label")
    top20_classes = sorted(
        counts, 
        key=counts.get, 
        reverse=True
    )[:20]
    detection_results.print_report(classes=top20_classes)
    non_empty_view = dataset.match(F("eval_tp") > 0)
    ## get true and false positive counts by image
    tp_vals = np.array(non_empty_view.values("eval_tp"))
    fp_vals = np.array(non_empty_view.values("eval_fp"))
    ## compute precision by image
    precision_vals = tp_vals/(tp_vals+fp_vals)
    ## set precision values
    non_empty_view.set_values("precision", precision_vals)
    ## get lowest precision images
    low_precision_view = non_empty_view.sort_by("precision")
    
    session = fo.launch_app(dataset)
    session.view = low_precision_view
    session.wait()

def show_dataset(dataset):
    
    session = fo.launch_app(dataset)
    session.wait()

def debug():
    path = 'runs/detect/predict/labels/000000000139.txt'
    output1=read_yolo_detections_file(path)
    output2=read_yolo_detections_file(path.replace("/","\\"))
    print(output2)

def get_train_dataset():
    train_dataset = foz.load_zoo_dataset(
        'coco-2017',
        split='train',
        classes=["potted plant"]
    ).clone()
    train_dataset.name = "potted_plants_train-data_new"
    train_dataset.persistent = True
    train_dataset.save()
    dataset_2007 = foz.load_zoo_dataset(
        "voc-2007",
        split="train",
        classes=["potted plant"]
    ).filter_labels(
        "ground_truth", 
        F("label") == "potted plant").clone()

    train_dataset.merge_samples(dataset_2007)
    return train_dataset

def prepare_training_data():
    train_dataset = get_train_dataset()
    train_dataset.untag_samples(train_dataset.distinct("tags"))
    four.random_split(
        train_dataset,
        {"train": 0.8, "val": 0.2}
    )
    export_yolo_data(
        train_dataset, 
        "potted_plant_train", 
        ["potted plant"], 
        split = ["train", "val"]
    )
    train_dataset.save()

def get_test_dataset() -> fo.Dataset:
    return fo.load_dataset("plants_test_dataset")

def generate_test_dataset() -> fo.Dataset:
    dataset = foz.load_zoo_dataset(
        'coco-2017',
        split='validation',
    )
    dataset = add_detection(dataset)

    test_dataset = dataset.filter_labels(
    "ground_truth", 
    F("label") == "potted plant"
    ).filter_labels(
        "yolov8n", 
        F("label") == "potted plant",
        only_matches=False
    ).clone()
    test_dataset.name = "plants_test_dataset"
    test_dataset.persistent = True
    
    base_results = test_dataset.evaluate_detections(
        "yolov8n", 
        eval_key="base",
        compute_mAP=True,
    )
    print(base_results.mAP())
    ## 0.24897924786479841
    base_results.print_report(classes=["potted plant"])
    export_yolo_data(
        test_dataset, 
        "potted_plant_test", 
        "potted plant"
    )
    test_dataset.save()

def add_detection_to_test(test_dataset, run_number=2, model_name="yolov8n"):
    filepaths = test_dataset.values("filepath")
    prediction_filepaths = [get_prediction_filepath(fp, run_number=run_number) for fp in filepaths]
    test_dataset.set_values(
        f"{model_name}_plant_det_filepath",
        prediction_filepaths
    )
    add_yolo_detections(
        test_dataset, 
        f"{model_name}", 
        f"{model_name}_plant_det_filepath", 
        ["potted plant"]
    )
    return test_dataset

if __name__ == "__main__":
    model_name = "yolov8m_plant_train"
    test_dataset = get_test_dataset()

    test_dataset = add_detection_to_test(test_dataset, run_number=6, model_name=model_name)
    finetune_plant_results = test_dataset.evaluate_detections(
        model_name, 
        eval_key=f"finetune_{model_name}",
        compute_mAP=True,
    )

    print("fine-tuned mAP: {}".format(finetune_plant_results.mAP()))

    finetune_plant_results.print_report(classes=["potted plant"])
    plot = finetune_plant_results.plot_pr_curves()
    plot.show()
    show_dataset(test_dataset)


    
    
