#!/usr/bin/env python3
"""
PromptFL with Flower - Docker Server
基于当前完整版本的Docker server实现
"""

import os
import sys
from pathlib import Path
import argparse
import logging
import time

# 设置环境变量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['FLWR_TELEMETRY_ENABLED'] = '0'

# 禁用TensorFlow相关的导入
sys.modules['tensorflow'] = None
sys.modules['tensorboard'] = None

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

import flwr as fl
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# 导入我们的组件
from config_flower import get_cfg_default, extend_cfg
from trainers.promptfl_flower import PromptFLFlower
from trainers.baseline import BaselineFlower
from trainers.coop import CoOpFlower
from utils_flower import average_weights

# Trainer registry
TRAINER_REGISTRY = {
    "PromptFL": PromptFLFlower,
    "Baseline": BaselineFlower,
    "CoOp": CoOpFlower,
}

def build_trainer(cfg):
    """Build trainer based on config"""
    trainer_name = cfg.TRAINER.NAME
    if trainer_name not in TRAINER_REGISTRY:
        raise ValueError(f"Unknown trainer: {trainer_name}. Available: {list(TRAINER_REGISTRY.keys())}")
    
    trainer_class = TRAINER_REGISTRY[trainer_name]
    return trainer_class(cfg)

class PromptFLStrategy(fl.server.strategy.FedAvg):
    """PromptFL联邦学习策略"""
    
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.trainer = build_trainer(cfg)
        
        # 初始化全局模型
        self.trainer.fed_before_train(is_global=True)
        
        super().__init__(**kwargs)
        
        print(f"🌸 PromptFL Strategy initialized")
        print(f"   Trainer: {cfg.TRAINER.NAME}")
        print(f"   Dataset: {cfg.DATASET.NAME}")
        print(f"   Backbone: {cfg.MODEL.BACKBONE.NAME}")
        print(f"   Users: {cfg.DATASET.USERS}")
        print(f"   Rounds: {cfg.OPTIM.ROUND}")
    
    def initialize_parameters(self, client_manager):
        """初始化全局参数"""
        # 获取初始参数
        initial_params = self.trainer.model.state_dict()
        
        # 转换为Flower参数格式
        params_list = []
        for key, value in initial_params.items():
            if 'prompt_learner' in key:  # 只传输prompt参数
                params_list.append(value.cpu().numpy())
        
        return fl.common.ndarrays_to_parameters(params_list)
    
    def aggregate_fit(self, server_round, results, failures):
        """聚合训练结果"""
        if not results:
            return None, {}
        
        print(f"\n🔄 Round {server_round}: Aggregating {len(results)} client updates")
        
        # 提取参数和指标
        weights_results = [
            (fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # 聚合参数
        aggregated_weights = self.aggregate_weights(weights_results)
        
        # 更新全局模型
        self.update_global_model(aggregated_weights)
        
        # 聚合指标
        metrics = {}
        if results:
            total_examples = sum([fit_res.num_examples for _, fit_res in results])
            
            # 计算加权平均指标
            for metric_name in ['train_loss', 'train_accuracy']:
                if all(metric_name in fit_res.metrics for _, fit_res in results):
                    weighted_sum = sum([
                        fit_res.metrics[metric_name] * fit_res.num_examples 
                        for _, fit_res in results
                    ])
                    metrics[metric_name] = weighted_sum / total_examples
        
        print(f"   📊 Aggregated metrics: {metrics}")
        
        return fl.common.ndarrays_to_parameters(aggregated_weights), metrics
    
    def aggregate_weights(self, weights_results):
        """聚合权重"""
        # 计算总样本数
        total_examples = sum([num_examples for _, num_examples in weights_results])
        
        # 加权平均
        aggregated = None
        for weights, num_examples in weights_results:
            weight = num_examples / total_examples
            
            if aggregated is None:
                aggregated = [w * weight for w in weights]
            else:
                for i, w in enumerate(weights):
                    aggregated[i] += w * weight
        
        return aggregated
    
    def update_global_model(self, aggregated_weights):
        """更新全局模型"""
        # 更新模型参数
        state_dict = self.trainer.model.state_dict()
        param_idx = 0
        
        for key, value in state_dict.items():
            if 'prompt_learner' in key:
                if param_idx < len(aggregated_weights):
                    state_dict[key] = torch.from_numpy(aggregated_weights[param_idx])
                    param_idx += 1
        
        self.trainer.model.load_state_dict(state_dict)
    
    def evaluate(self, server_round, parameters):
        """服务器端评估"""
        print(f"\n🔍 Round {server_round}: Server evaluation")
        
        # 更新模型参数
        if parameters:
            aggregated_weights = fl.common.parameters_to_ndarrays(parameters)
            self.update_global_model(aggregated_weights)
        
        # 执行评估
        try:
            accuracy, error_rate, f1_score = self.trainer.test(is_global=True, current_epoch=server_round)
            
            metrics = {
                "accuracy": accuracy,
                "error_rate": error_rate,
                "f1_score": f1_score
            }
            
            print(f"   📈 Server metrics: {metrics}")
            return 0.0, metrics  # loss, metrics
            
        except Exception as e:
            print(f"   ❌ Server evaluation failed: {e}")
            return 0.0, {}

def create_config(args):
    """创建配置"""
    cfg = get_cfg_default()
    extend_cfg(cfg)
    
    # 设置基本配置
    cfg.TRAINER.NAME = args.trainer
    cfg.DATASET.NAME = args.dataset
    cfg.DATASET.USERS = args.num_clients
    cfg.DATASET.IID = args.iid
    cfg.MODEL.BACKBONE.NAME = args.backbone
    cfg.OPTIM.ROUND = args.rounds
    cfg.OPTIM.MAX_EPOCH = args.local_epochs
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.batch_size
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.USE_CUDA = False
    
    # Trainer特定配置
    if args.trainer == "PromptFL":
        cfg.TRAINER.PROMPTFL.N_CTX = args.n_ctx
        cfg.TRAINER.PROMPTFL.PREC = "fp32"
    elif args.trainer == "CoOp":
        cfg.TRAINER.COOP.N_CTX = args.n_ctx
        cfg.TRAINER.COOP.PREC = "fp32"
    elif args.trainer == "Baseline":
        cfg.TRAINER.BASELINE.PREC = "fp32"
    
    cfg.freeze()
    return cfg

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PromptFL with Flower - Docker Server")
    
    # 服务器配置
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds")
    parser.add_argument("--min-clients", type=int, default=2, help="Minimum number of clients")
    
    # 模型配置
    parser.add_argument("--trainer", type=str, default="PromptFL", 
                       choices=["PromptFL", "CoOp", "Baseline"], help="Trainer type")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name")
    parser.add_argument("--backbone", type=str, default="ViT-B/16", help="CLIP backbone")
    parser.add_argument("--n-ctx", type=int, default=16, help="Number of context tokens")
    
    # 训练配置
    parser.add_argument("--num-clients", type=int, default=2, help="Number of clients")
    parser.add_argument("--local-epochs", type=int, default=5, help="Local epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--iid", action="store_true", help="IID data distribution")
    
    args = parser.parse_args()
    
    print("🌸 PromptFL with Flower - Docker Server")
    print("=" * 50)
    print(f"Host: {args.host}:{args.port}")
    print(f"Trainer: {args.trainer}")
    print(f"Dataset: {args.dataset}")
    print(f"Backbone: {args.backbone}")
    print(f"Rounds: {args.rounds}")
    print(f"Min Clients: {args.min_clients}")
    
    # 创建配置
    cfg = create_config(args)
    
    # 创建策略
    strategy = PromptFLStrategy(
        cfg=cfg,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
    )
    
    # 启动服务器
    print(f"\n🚀 Starting Flower server on {args.host}:{args.port}")
    
    fl.server.start_server(
        server_address=f"{args.host}:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()