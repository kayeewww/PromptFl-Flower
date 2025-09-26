#!/usr/bin/env python3
"""
Data Manager for PromptFL with Flower
基于Dassl数据管理器的简化版本，专门用于Flower联邦学习
"""

import torch
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import sys
import os

# 添加项目路径
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_root)

# 动态导入数据集
def import_dataset_class(dataset_name):
    """动态导入数据集类"""
    try:
        if dataset_name == "cifar10":
            from datasets.cifar10 import CIFAR10Dataset
            return CIFAR10Dataset
        elif dataset_name == "caltech101":
            from datasets.caltech101 import Caltech101Dataset
            return Caltech101Dataset
        elif dataset_name == "food101":
            from datasets.food101 import Food101Dataset
            return Food101Dataset
        elif dataset_name == "oxford_flowers":
            from datasets.oxford_flowers import OxfordFlowersDataset
            return OxfordFlowersDataset
        elif dataset_name == "oxford_pets":
            from datasets.oxford_pets import OxfordPetsDataset
            return OxfordPetsDataset
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    except ImportError as e:
        print(f"Failed to import dataset {dataset_name}: {e}")
        # 回退到简单的数据集实现
        return SimpleCIFAR10Dataset
from sampling_flower import cifar_iid, cifar_noniid

class DataManagerFlower:
    """简化的数据管理器，专门用于Flower联邦学习"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        
        # 构建数据集
        self.build_dataset()
        
        # 构建数据加载器
        self.build_data_loaders()
        
        # 分割联邦数据
        self.build_federated_data()
        
    def build_dataset(self):
        """构建数据集"""
        cfg = self.cfg
        dataset_name = cfg.DATASET.NAME.lower()
        
        # 动态导入数据集类
        DatasetClass = import_dataset_class(dataset_name)
        self.dataset = DatasetClass(cfg)
            
        print(f"Dataset: {dataset_name}")
        print(f"Number of classes: {self.dataset.num_classes}")
        print(f"Training samples: {len(self.dataset.train_x)}")
        print(f"Test samples: {len(self.dataset.test)}")
        
    def build_data_loaders(self):
        """构建数据加载器"""
        cfg = self.cfg
        
        # 训练数据加载器
        self.train_loader = DataLoader(
            self.dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=True
        )
        
        # 测试数据加载器
        self.test_loader = DataLoader(
            self.dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=False
        )
        
    def build_federated_data(self):
        """构建联邦学习数据分割"""
        cfg = self.cfg
        
        # 根据IID设置选择数据分割方法
        if cfg.DATASET.IID:
            if cfg.DATASET.NAME.lower() == "cifar10":
                user_groups = cifar_iid(self.dataset.train_x, cfg.DATASET.USERS)
            else:
                # 对其他数据集使用简单的IID分割
                user_groups = self._simple_iid_split(self.dataset.train_x, cfg.DATASET.USERS)
        else:
            if cfg.DATASET.NAME.lower() == "cifar10":
                user_groups = cifar_noniid(self.dataset.train_x, cfg.DATASET.USERS)
            else:
                # 对其他数据集使用简单的非IID分割
                user_groups = self._simple_noniid_split(self.dataset.train_x, cfg.DATASET.USERS)
        
        # 创建客户端数据加载器
        self.client_loaders = {}
        for client_id in range(cfg.DATASET.USERS):
            indices = user_groups[client_id]
            client_dataset = Subset(self.dataset.train_x, indices)
            
            client_loader = DataLoader(
                client_dataset,
                batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                shuffle=True,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                drop_last=True
            )
            
            self.client_loaders[client_id] = client_loader
            
            print(f"Client {client_id}: {len(indices)} samples")
            
    def _simple_iid_split(self, dataset, num_users):
        """简单的IID数据分割"""
        import numpy as np
        
        num_items = len(dataset) // num_users
        dict_users = {}
        all_idxs = list(range(len(dataset)))
        
        for i in range(num_users):
            selected_idxs = set(np.random.choice(all_idxs, num_items, replace=False))
            dict_users[i] = list(selected_idxs)
            all_idxs = list(set(all_idxs) - selected_idxs)
            
        return dict_users
        
    def _simple_noniid_split(self, dataset, num_users):
        """简单的非IID数据分割（基于类别）"""
        import numpy as np
        from collections import defaultdict
        
        # 按类别分组
        class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            class_indices[label].append(idx)
        
        # 为每个客户端分配类别
        num_classes = len(class_indices)
        classes_per_client = max(1, num_classes // num_users)
        
        dict_users = {i: [] for i in range(num_users)}
        class_list = list(class_indices.keys())
        np.random.shuffle(class_list)
        
        for i in range(num_users):
            # 为每个客户端分配几个类别
            start_idx = (i * classes_per_client) % num_classes
            end_idx = min(start_idx + classes_per_client, num_classes)
            
            client_classes = class_list[start_idx:end_idx]
            if len(client_classes) == 0:
                client_classes = [class_list[i % num_classes]]
            
            # 收集这些类别的所有样本
            for class_id in client_classes:
                dict_users[i].extend(class_indices[class_id])
        
        return dict_users
        
    def get_client_data(self, client_id):
        """获取指定客户端的数据加载器"""
        if client_id not in self.client_loaders:
            raise ValueError(f"Client {client_id} not found")
        return self.client_loaders[client_id]
        
    @property
    def num_classes(self):
        """返回类别数量"""
        return self.dataset.num_classes
        
    @property
    def lab2cname(self):
        """返回标签到类名的映射"""
        return getattr(self.dataset, 'lab2cname', {})
        
    def show_dataset_summary(self):
        """显示数据集摘要"""
        cfg = self.cfg
        
        print("Dataset summary:")
        print(f"  Dataset: {cfg.DATASET.NAME}")
        print(f"  Number of classes: {self.num_classes}")
        print(f"  Number of clients: {cfg.DATASET.USERS}")
        print(f"  IID: {cfg.DATASET.IID}")
        print(f"  Training samples: {len(self.dataset.train_x)}")
        print(f"  Test samples: {len(self.dataset.test)}")
        
        # 显示每个客户端的数据分布
        for client_id in range(cfg.DATASET.USERS):
            client_size = len(self.client_loaders[client_id].dataset)
            print(f"  Client {client_id}: {client_size} samples")


class SimpleCIFAR10Dataset:
    """简单的CIFAR-10数据集实现（回退选项）"""
    
    def __init__(self, cfg):
        import torchvision
        import torchvision.transforms as transforms
        from torch.utils.data import Dataset
        
        self.cfg = cfg
        self.root = cfg.DATASET.ROOT or "./data"
        
        # 定义变换
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载数据集
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.root, train=True, download=True, transform=transform_train
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.root, train=False, download=True, transform=transform_test
        )
        
        self.train_x = train_dataset
        self.test = test_dataset
        
        # 设置类名和其他属性
        self.classnames = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        self.lab2cname = {i: name for i, name in enumerate(self.classnames)}
        self.num_classes = len(self.classnames)
        
        print(f"Simple CIFAR-10 loaded:")
        print(f"  Training samples: {len(self.train_x)}")
        print(f"  Test samples: {len(self.test)}")