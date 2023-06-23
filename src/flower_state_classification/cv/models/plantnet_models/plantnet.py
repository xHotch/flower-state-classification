from collections import defaultdict
import gc
import json
import os
import torch
import timm
import cv2
import numpy as np
import importlib.resources as pkg_resources
from torchvision.models import resnet18
from flower_state_classification.cv.models.modeltypes import Classifier
from flower_state_classification.util.benchmark import benchmark_fps


class PlantNet(Classifier):
    """
    PlantNet is a classifier that can classify a plant image into one of 1081 different species.
    The source code for the model can be found here: https://github.com/plantnet/PlantNet-300K
    There are pretrained weights for different models, that can be downloaded here: https://lab.plantnet.org/seafile/d/01ab6658dad6447c95ae/
    
    The downloaded models should be placed in the same folder as this file.

    Args:
        model (str, optional): The model to use. Defaults to "vit_base_patch16_224".
        use_gpu (bool, optional): Whether to use the GPU or not. Defaults to False.
        return_genus (bool, optional): Whether to return the genus of the plant or the full species name. Defaults to False.
    """
    model: torch.nn.Module
    species_id_to_species_name: dict
    idx_to_species_id: dict
    genus_to_species_id: dict


    def __init__(self, model_name: str = "vit_base_patch16_224", use_gpu = False, return_genus = False) -> None:
        super().__init__()
        self.dir = os.path.dirname(os.path.abspath(__file__))
        self.model_name = model_name
        if model_name == "vit_base_patch16_224":
            self.model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=1081)
            self.weight_file = os.path.join(self.dir,"vit_base_patch16_224_weights_best_acc.tar")
        else:
            self.model = resnet18(num_classes=1081)
            self.weight_file = os.path.join(self.dir,"resnet18_weights_best_acc.tar")

        self.image_size = (224, 224)
        self.use_gpu = use_gpu
        self.return_genus = return_genus
        self._load_classmap()
        self._load_model()

    def _load_model(self) -> None:
        self.model = self.model.cuda() if self.use_gpu else self.model.cpu()
        device = 'cuda:0' if self.use_gpu else 'cpu'
        d = torch.load(self.weight_file, map_location=device)
        self.model.load_state_dict(d['model'])
        self.model.eval()
    
    def _load_classmap(self) -> None:
        with open(os.path.join(self.dir ,"idx_2_species_id.json"),"r") as f:
            self.idx_to_species_id = json.load(f)
        with open(os.path.join(self.dir ,"species_id_2_species_name.json"),"r") as f:
            self.species_id_to_species_name = json.load(f)
        self.genus_to_species_id = defaultdict(set)
        for species, species_id in self.species_id_to_species_name.items():
            genus = species.split(" ")[0]
            self.genus_to_species_id[genus].add(species_id)


    def _species_id_to_genus(self, species_id: str) -> str:
        for key, value in self.genus_to_species_id.items():
            if species_id in value:
                return key

    @benchmark_fps(cooldown = 1)
    def predict(self, plant_frame: np.array) -> str:
        frame_resized = cv2.resize(plant_frame, self.image_size)
        frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).unsqueeze(0).float()
        
        frame_tensor = frame_tensor.cuda() if self.use_gpu else frame_tensor.cpu()
        with torch.no_grad():
            output = self.model(frame_tensor)
            _, predicted = torch.max(output.data, 1)
            idx = str(predicted.item())
        del frame_tensor
        del frame_resized
        del output
        if self.return_genus:
            return self._species_id_to_genus(self.idx_to_species_id[idx])
        else:
            return self.species_id_to_species_name[self.idx_to_species_id[idx]]

    def __str__(self):
        return f"PlantNet({self.model_name})"