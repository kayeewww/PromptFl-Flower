#!/usr/bin/env python3
"""
Baseline Trainer for PromptFL with Flower - Full model fine-tuning
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

import clip
from utils_flower import AverageMeter, count_num_param

class BaselineFlower:
    """Baseline trainer - full model fine-tuning"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() and cfg.USE_CUDA else "cpu")
        
        # 构建数据管理器
        self.build_data_manager()
        
        # 构建模型
        self.build_model()
        
        # 构建优化器
        self.build_optimizer()
        
        # 其他初始化
        self.scaler = GradScaler() if cfg.TRAINER.BASELINE.PREC == "amp" else None
        
    def build_data_manager(self):
        """构建数据管理器"""
        from data_manager_flower import DataManagerFlower
        self.dm = DataManagerFlower(self.cfg)
        
    def build_model(self):
        """构建模型 - 使用完整的CLIP模型进行微调"""
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP for baseline fine-tuning (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model, _ = clip.load(cfg.MODEL.BACKBONE.NAME, device=self.device)
        
        if cfg.TRAINER.BASELINE.PREC == "fp32":
            clip_model.float()

        # 创建分类头
        self.model = BaselineCLIP(clip_model, len(classnames))
        
        print("Fine-tuning entire CLIP model")
        print(f"# total params: {count_num_param(self.model):,}")

        self.model.to(self.device)
        
    def build_optimizer(self):
        """构建优化器 - 优化整个模型"""
        cfg = self.cfg
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.OPTIM.LR,
            weight_decay=cfg.OPTIM.WEIGHT_DECAY
        )
        
    def fed_before_train(self, is_global=False):
        """联邦学习训练前的准备"""
        pass
        
    def fed_after_train(self):
        """联邦学习训练后的清理"""
        pass
        
    def train(self, idx=None, global_epoch=None, is_fed=False):
        """训练函数"""
        cfg = self.cfg
        self.model.train()
        
        # 获取客户端数据
        if is_fed and idx is not None:
            train_loader = self.dm.get_client_data(idx)
        else:
            train_loader = self.dm.train_loader
            
        losses = AverageMeter()
        accuracies = AverageMeter()
        
        for epoch in range(cfg.OPTIM.MAX_EPOCH):
            for batch_idx, batch in enumerate(train_loader):
                loss_summary = self.forward_backward(batch)
                losses.update(loss_summary["loss"])
                accuracies.update(loss_summary["acc"])
                
        print(f"Baseline Client {idx} - Epoch {global_epoch}: Loss={losses.avg:.4f}, Acc={accuracies.avg:.4f}")
        
    def forward_backward(self, batch):
        """前向和反向传播"""
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.BASELINE.PREC
        
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            if output.dtype == torch.float16:
                output = output.float()
            loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        # 计算准确率
        _, predicted = torch.max(output, 1)
        accuracy = (predicted == label).float().mean()

        loss_summary = {
            "loss": loss.item(),
            "acc": accuracy.item(),
        }

        return loss_summary

    def parse_batch_train(self, batch):
        """解析训练批次"""
        if isinstance(batch, dict):
            input = batch["img"]
            label = batch["label"]
        else:
            input, label = batch
            
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
        
    def test(self, is_global=False, current_epoch=None):
        """测试函数"""
        self.model.eval()
        
        test_loader = self.dm.test_loader
        total_correct = 0
        total_samples = 0
        total_loss = 0
        
        with torch.no_grad():
            for batch in test_loader:
                image, label = self.parse_batch_train(batch)
                output = self.model(image)
                loss = F.cross_entropy(output, label)
                
                _, predicted = torch.max(output, 1)
                total_correct += (predicted == label).sum().item()
                total_samples += label.size(0)
                total_loss += loss.item()
        
        accuracy = total_correct / total_samples
        avg_loss = total_loss / len(test_loader)
        error_rate = 1 - accuracy
        
        print(f"Baseline Test - Epoch {current_epoch}: Acc={accuracy:.4f}, Loss={avg_loss:.4f}")
        
        return accuracy, error_rate, accuracy

class BaselineCLIP(nn.Module):
    """Baseline CLIP model with classification head"""
    
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip_model = clip_model
        self.classifier = nn.Linear(clip_model.visual.output_dim, num_classes)
        
    def forward(self, image):
        # Extract image features using CLIP
        image_features = self.clip_model.encode_image(image)
        
        # Apply classification head
        logits = self.classifier(image_features)
        
        return logits