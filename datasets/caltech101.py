#!/usr/bin/env python3
"""
Caltech101 dataset for PromptFL with Flower
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import Caltech101 as TorchCaltech101

class Caltech101Dataset:
    """Caltech101 dataset wrapper for PromptFL"""
    
    def __init__(self, root="./data", transform=None):
        self.root = root
        self.transform = transform or self._get_default_transform()
        
        # Load dataset
        self.dataset = TorchCaltech101(
            root=root, 
            download=True,
            transform=self.transform
        )
        
        # Get class names
        self.classnames = [
            'accordion', 'airplanes', 'anchor', 'ant', 'barrel', 'bass', 'beaver', 'binocular',
            'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera', 'cannon',
            'car_side', 'ceiling_fan', 'cellphone', 'chair', 'chandelier', 'cougar_body',
            'cougar_face', 'crab', 'crayfish', 'crocodile', 'crocodile_head', 'cup', 'dalmatian',
            'dollar_bill', 'dolphin', 'dragonfly', 'electric_guitar', 'elephant', 'emu', 'euphonium',
            'ewer', 'faces', 'faces_easy', 'ferry', 'flamingo', 'flamingo_head', 'garfield',
            'gerenuk', 'gramophone', 'grand_piano', 'hawksbill', 'headphone', 'hedgehog',
            'helicopter', 'ibis', 'inline_skate', 'joshua_tree', 'kangaroo', 'ketch', 'lamp',
            'laptop', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome',
            'minaret', 'motorbikes', 'nautilus', 'octopus', 'okapi', 'pagoda', 'panda',
            'pigeon', 'pizza', 'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone',
            'schooner', 'scissors', 'scorpion', 'sea_horse', 'snoopy', 'soccer_ball', 'stapler',
            'starfish', 'stegosaurus', 'stop_sign', 'strawberry', 'sunflower', 'tick', 'trilobite',
            'umbrella', 'watch', 'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair',
            'wrench', 'yin_yang'
        ]
        
        self.name = "caltech101"
        self.num_classes = len(self.classnames)
        
    def _get_default_transform(self):
        """Get default data transforms"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]