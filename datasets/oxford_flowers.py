#!/usr/bin/env python3
"""
Oxford Flowers dataset for PromptFL with Flower
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import Flowers102

class OxfordFlowersDataset:
    """Oxford Flowers dataset wrapper for PromptFL"""
    
    def __init__(self, root="./data", transform=None):
        self.root = root
        self.transform = transform or self._get_default_transform()
        
        # Load dataset
        self.train_dataset = Flowers102(
            root=root, 
            split='train',
            download=True,
            transform=self.transform
        )
        
        self.test_dataset = Flowers102(
            root=root, 
            split='test',
            download=True,
            transform=self.transform
        )
        
        # Get class names (simplified flower names)
        self.classnames = [
            'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',
            'english marigold', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood',
            'globe thistle', 'snapdragon', 'colt\'s foot', 'king protea', 'spear thistle',
            'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower',
            'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary',
            'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers',
            'stemless gentian', 'artichoke', 'sweet william', 'carnation', 'garden phlox',
            'love in the mist', 'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya',
            'cape flower', 'great masterwort', 'siam tulip', 'lenten rose', 'barbeton daisy',
            'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue', 'wallflower',
            'marigold', 'buttercup', 'oxeye daisy', 'common dandelion', 'petunia',
            'wild pansy', 'primula', 'sunflower', 'pelargonium', 'bishop of llandaff',
            'gaura', 'geranium', 'orange dahlia', 'pink-yellow dahlia', 'cautleya spicata',
            'japanese anemone', 'black-eyed susan', 'silverbush', 'californian poppy',
            'osteospermum', 'spring crocus', 'bearded iris', 'windflower', 'tree poppy',
            'gazania', 'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory',
            'passion flower', 'lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis',
            'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen',
            'watercress', 'canna lily', 'hippeastrum', 'bee balm', 'ball moss', 'foxglove',
            'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower',
            'trumpet creeper', 'blackberry lily'
        ]
        
        self.name = "oxford_flowers"
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