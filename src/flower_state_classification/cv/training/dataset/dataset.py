from enum import Enum
import os
from typing import ClassVar, Dict, List

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

import logging

logger = logging.getLogger(__name__)

class ExportTypes(Enum):
    YOLO = fo.types.YOLOv5Dataset
    COCO = fo.types.COCODetectionDataset
    
class PottedPlantDataset():
    """
    Class that handles the potted plant dataset.
    """
    
    datasets: ClassVar[Dict[str, List[str]]] = {
        "coco-plants" : ["coco-2017"],
        "coco-voc-plants" : ["coco-2017", "voc-2007", "voc-2012"],
        "coco-voc-open_images-plants_v3" : ["coco-2017", "voc-2007", "voc-2012", "open-images-v7"]
    }

    classes: ClassVar[List[str]] = ["potted plant"]
    dataset: fo.Dataset

    def __init__(self, dataset_name: str, re_setup: bool = False):
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset name must be one of {self.datasets}")
        self.dataset_name = dataset_name
        self.export_folder = os.path.join(os.path.dirname(__file__), self.dataset_name)
        try:
            self.dataset = fo.load_dataset(dataset_name)
            if len(self.dataset) == 0 or re_setup:
                self._setup_dataset()
        except ValueError:
            logger.info("Creating Dataset")
            self.dataset = fo.Dataset()
            self.dataset.name = self.dataset_name
            self.dataset.persistent = True
            self._setup_dataset()
    
    def _setup_dataset(self):
        for dataset in self.datasets[self.dataset_name]:
            logger.info(f"Merging fiftyone dataset {dataset}. Number of samples {len(self.dataset)}")
            self.dataset.merge_samples(self.download_dataset(dataset).clone())
            logger.info(f"Number of samples after merge {len(self.dataset)}")
        self.dataset = self.dataset.filter_labels(
        "ground_truth", 
        F("label") == "potted plant",only_matches=True)
        self.dataset.save()
        
    def download_dataset(self, dataset_name: str):
        if dataset_name.startswith("open-images"):
            dataset = foz.load_zoo_dataset(
                dataset_name,
                label_types=["detections"],
                classes=["Houseplant"]
            ).map_labels("ground_truth", {"Houseplant": "potted plant"}) # Rename class for open images
            return dataset
        
        dataset = foz.load_zoo_dataset(
            dataset_name,
            label_types=["detections"],
            classes=self.classes
        )
        return dataset
    
    def export(self, export_type: ExportTypes, split: str = "train"):
        folder = self.get_export_folder(export_type, split = split)
        self.export_data(self.dataset, folder, "ground_truth", split, export_type.value)
    
    def get_fo_dataset(self):
        return self.dataset

    def export_data(self,
            samples, 
            export_dir, 
            label_field = "ground_truth", 
            split = "val",
            dataset_type = fo.types.YOLOv5Dataset
            ):
        if type(split) == list:
            splits = split
            for split in splits:
                self.export_data(
                    samples, 
                    export_dir,
                    label_field, 
                    split,
                    dataset_type
                )   
        else:
            split_view = samples.match_tags(split)
            split_view.export(
                export_dir=export_dir,
                dataset_type=dataset_type,
                label_field=label_field,
                classes=self.classes,
                split=split,
            )

    def get_export_folder(self, export_type: ExportTypes = None, split = None):
        if export_type is None:
            return self.export_folder
        export_path = os.path.join(self.export_folder, export_type.name.lower())
        if split is not None:
            export_path = os.path.join(export_path, split)
        return export_path

    def get_test_train_split(self, split: float = 0.8, max_samples: int = None):
        if max_samples:
            dataset = dataset.take(max_samples, seed=51)
        len_dataset = len(dataset)
        train_view = dataset.take(int(len_dataset*split), seed=51)
        test_view = dataset.exclude([s.id for s in train_view])

        return train_view, test_view
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataset = PottedPlantDataset("coco-plants", re_setup= True)
    # self.export(ExportTypes.YOLO)
    # dataset.export(ExportTypes.YOLO, "train")
    # dataset.export(ExportTypes.YOLO, "validation")
    # dataset.export(ExportTypes.COCO, "validation")
    # dataset.export(ExportTypes.YOLO, ["test", "validation"])

    #show dataset in app
    # session = fo.launch_app(dataset.get_fo_dataset())
    # session.wait()