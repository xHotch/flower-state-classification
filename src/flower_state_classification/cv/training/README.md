# Training Process
This file presents a short overview of the training process for the different deep learning frameworks. Most of the Code is based on the original implementation of the frameworks or tutorials found on the internet. For more indepth information, please refer to the original sources.

# Dataset
To handle the datasets for the different frameworks, we created a single dataset managed by the fiftyone python library. This allows us to export the actual images using different formats, and to visualize the results of the different training and evaluation runs.

The dataset can be found under dataset/dataset.py

# Model Training

## PyTorch
Deprecated, use ultralytics or PaddleDetection instead.

## Ultralytics
For ultralytics training, refer to https://docs.ultralytics.com/modes/train/. Before training, make sure the desired dataset is exported in the YOLO format. Afterwards simply install the ultralytics library using:

```pip install ultralytics```

No further code is needed.

The trained models will be saved in the ```runs/train{nr_training}``` folder. The training process can be monitored using tensorboard. To start tensorboard, use:

```tensorboard --logdir runs/train{nr_training}```

### Example Training Commands

```yolo task=detect mode=train model=yolov8m.pt data=../dataset/coco-voc-open_images-plants_v3/yolo/train/dataset.yaml epochs=90 imgsz=640 batch=12 lr0=0.001 optimizer=AdamW```

```yolo train resume model=runs/train14/last.pt```

## PaddleDetection
For training PaddlePaddle models using the PaddleDetection framework, we used the code provided under https://github.com/PaddlePaddle/PaddleDetection/tree/develop.
The code was modified to support the training our models by adding a configuration file for our dataset, as well as a configuration file for each model type we want to train.

The configuration files for the dataset can be found under ```configs/dataset/coco/plant_detection```. Before training, make sure the desired dataset is exported in the COCO format.

The configuration files for the model can be found under ```configs/rtdetr/rtdetr_hgnetv2_l_6x_plants.yml```