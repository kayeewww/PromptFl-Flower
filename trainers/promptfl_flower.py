#!/usr/bin/env python3
"""
PromptFL Trainer using Flower - 替代Dassl的TrainerX
"""

import os.path as osp
import os
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

# 使用根目录的clip模块
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from utils_flower import MetricMeter, AverageMeter, count_num_param
from sampling_flower import cifar_iid, cifar_noniid

_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.PROMPTFL.N_CTX
        ctx_init = cfg.TRAINER.PROMPTFL.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            if cfg.TRAINER.PROMPTFL.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.PROMPTFL.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError

        return prompts

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

class PromptFLFlower:
    """PromptFL Trainer using Flower instead of Dassl"""
    
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
        self.scaler = GradScaler() if cfg.TRAINER.PROMPTFL.PREC == "amp" else None
        
    def build_data_manager(self):
        """构建数据管理器"""
        from data_manager_flower import DataManagerFlower
        self.dm = DataManagerFlower(self.cfg)
        
    def build_model(self):
        """构建模型"""
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PROMPTFL.PREC == "fp32" or cfg.TRAINER.PROMPTFL.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
                
        print(f"# params: {count_num_param(self.model):,}")
        print(f"# prompt learner params: {count_num_param(self.model.prompt_learner):,}")

        self.model.to(self.device)
        
    def build_optimizer(self):
        """构建优化器"""
        cfg = self.cfg
        # 只优化prompt_learner
        self.optim = torch.optim.Adam(
            self.model.prompt_learner.parameters(),
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
                
        print(f"Client {idx} - Epoch {global_epoch}: Loss={losses.avg:.4f}, Acc={accuracies.avg:.4f}")
        
    def forward_backward(self, batch):
        """前向和反向传播"""
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.PROMPTFL.PREC
        
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
            # 确保输出是float32类型以避免半精度问题
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
        
        print(f"Test - Epoch {current_epoch}: Acc={accuracy:.4f}, Loss={avg_loss:.4f}")
        
        # 返回 (accuracy, error_rate, f1_score)
        return accuracy, error_rate, accuracy  # 简化版本，用accuracy代替f1_score