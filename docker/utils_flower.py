#!/usr/bin/env python3
"""
PromptFL with Flower - Utilities
基于Dassl工具函数的Flower版本实现
"""

import copy
from collections import defaultdict
import torch
import numpy as np
from typing import Dict, List, Any

__all__ = ["AverageMeter", "MetricMeter", "count_num_param", "average_weights"]


class AverageMeter:
    """Compute and store the average and current value.

    Examples::
        >>> # 1. Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # 2. Update meter after every mini-batch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self, ema=False):
        """
        Args:
            ema (bool, optional): apply exponential moving average.
        """
        self.ema = ema
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()

        self.val = val
        self.sum += val * n
        self.count += n

        if self.ema:
            self.avg = self.avg * 0.9 + self.val * 0.1
        else:
            self.avg = self.sum / self.count


class MetricMeter:
    """Store the average and current value for a set of metrics.

    Examples::
        >>> # 1. Create an instance of MetricMeter
        >>> metric = MetricMeter()
        >>> # 2. Update using a dictionary as input
        >>> input_dict = {'loss_1': value_1, 'loss_2': value_2}
        >>> metric.update(input_dict)
        >>> # 3. Convert to string and print
        >>> print(str(metric))
    """

    def __init__(self, delimiter=" "):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, input_dict):
        if input_dict is None:
            return

        if not isinstance(input_dict, dict):
            print("input_dict", input_dict)
            raise TypeError(
                "Input to MetricMeter.update() must be a dictionary"
            )

        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def __str__(self):
        output_str = []
        for name, meter in self.meters.items():
            output_str.append(f"{name} {meter.val:.4f} ({meter.avg:.4f})")
        return self.delimiter.join(output_str)


def count_num_param(model=None, params=None):
    """Count number of parameters in a model.

    Args:
        model (nn.Module): network model.
        params: network model's params.
        
    Examples::
        >>> model_size = count_num_param(model)
    """
    if model is not None:
        return sum(p.numel() for p in model.parameters())

    if params is not None:
        s = 0
        for p in params:
            if isinstance(p, dict):
                s += p["params"].numel()
            else:
                s += p.numel()
        return s


def average_weights(w_list: List[List[np.ndarray]], weights: List[float] = None) -> List[np.ndarray]:
    """
    Average model weights from multiple clients.
    
    Args:
        w_list: List of parameter lists from different clients
        weights: List of weights for each client (e.g., based on data size)
                If None, simple average is used
    
    Returns:
        List of averaged parameters
    """
    if not w_list:
        return []
    
    if weights is None:
        # Simple average
        weights = [1.0 / len(w_list)] * len(w_list)
    else:
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
    
    # Initialize averaged weights with zeros
    num_params = len(w_list[0])
    averaged_weights = []
    
    for param_idx in range(num_params):
        # Get the shape from the first client
        param_shape = w_list[0][param_idx].shape
        averaged_param = np.zeros(param_shape)
        
        # Weighted average across all clients
        for client_idx, client_weights in enumerate(w_list):
            averaged_param += weights[client_idx] * client_weights[param_idx]
        
        averaged_weights.append(averaged_param)
    
    return averaged_weights


def aggregate_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate metrics from multiple clients.
    
    Args:
        metrics_list: List of metric dictionaries from different clients
    
    Returns:
        Dictionary with aggregated metrics
    """
    if not metrics_list:
        return {}
    
    aggregated = {}
    
    # Get all metric keys
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())
    
    for key in all_keys:
        values = []
        for metrics in metrics_list:
            if key in metrics:
                values.append(metrics[key])
        
        if values:
            if isinstance(values[0], (int, float)):
                # Numerical values - compute average
                aggregated[key] = sum(values) / len(values)
            else:
                # Non-numerical values - take the first one or create a list
                aggregated[key] = values[0] if len(values) == 1 else values
    
    return aggregated


def format_metrics(metrics: Dict[str, Any], prefix: str = "") -> str:
    """
    Format metrics dictionary into a readable string.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Optional prefix for the output string
    
    Returns:
        Formatted string
    """
    if not metrics:
        return f"{prefix}No metrics available"
    
    formatted_items = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted_items.append(f"{key}: {value:.4f}")
        else:
            formatted_items.append(f"{key}: {value}")
    
    return f"{prefix}{', '.join(formatted_items)}"


def save_metrics_to_file(metrics: Dict[str, Any], filepath: str):
    """
    Save metrics to a file.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save the metrics
    """
    import json
    
    # Convert numpy arrays and tensors to lists for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        elif isinstance(value, torch.Tensor):
            serializable_metrics[key] = value.cpu().numpy().tolist()
        else:
            serializable_metrics[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)


def load_metrics_from_file(filepath: str) -> Dict[str, Any]:
    """
    Load metrics from a file.
    
    Args:
        filepath: Path to load the metrics from
    
    Returns:
        Dictionary of metrics
    """
    import json
    
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    
    return metrics