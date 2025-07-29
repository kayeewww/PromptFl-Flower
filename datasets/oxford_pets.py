#!/usr/bin/env python3
"""
Oxford Pets dataset for PromptFL with Flower
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet

class OxfordPetsDataset:
    """Oxford Pets dataset wrapper for PromptFL"""
    
    def __init__(self, root="./data", transform=None):
        self.root = root
        self.transform = transform or self._get_default_transform()
        
        # Load dataset
        self.train_dataset = OxfordIIITPet(
            root=root, 
            split='trainval',
            download=True,
            transform=self.transform
        )
        
        self.test_dataset = OxfordIIITPet(
            root=root, 
            split='test',
            download=True,
            transform=self.transform
        )
        
        # Get class names
        self.classnames = [
            'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau',
            'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx',
            'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle',
            'boxer', 'chihuahua', 'english_cocker_spaniel', 'english_setter',
            'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin',
            'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland',
            'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier',
            'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier'
        ]
        
        self.name = "oxford_pets"
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
    
    def get_train_data(self):
        return self.train_dataset
    
    def get_test_data(self):
        return self.test_dataset