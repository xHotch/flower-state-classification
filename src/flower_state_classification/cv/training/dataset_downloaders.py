"""
Use this script to download different datasets for training and testing.
"""

import fiftyone as fo
import fiftyone.zoo as foz

def get_coco_dataset(*args, **kwargs):
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        label_types=["detections"],
        classes=["potted plant"]
    )
    dataset.persistent = True
    return dataset    

def get_voc_datasets():
    dataset_2007 = foz.load_zoo_dataset(
        "voc-2007",
        label_types=["detections"],
        classes=["potted plant"]
    )
    dataset_2007.persistent = True

    dataset_2012 = foz.load_zoo_dataset(
        "voc-2012",
        label_types=["detections"],
        classes=["potted plant"]
    )
    dataset_2012.persistent = True

    return dataset_2007, dataset_2012

def list_detection_datasets():
    available_datasets = foz.list_zoo_datasets()
    for dataset in available_datasets:
        dataset_foz = foz.get_zoo_dataset(dataset)
        if "detection" in dataset_foz.tags:
            print(dataset)
            print(dataset_foz)

if __name__ == "__main__":
    print(fo.config.dataset_zoo_dir)
    downloaded_datasets = foz.list_downloaded_zoo_datasets()
    fo.pprint(downloaded_datasets)
    list_detection_datasets()
    dataset = get_voc_datasets()
    #session = fo.launch_app(dataset)
    #session.wait()
