import torch
import torch.nn.functional as F

def binary_focal_loss(inputs, targets, alpha=1, gamma=2, **kwargs):
    "based on https://medium.com/data-scientists-diary/implementing-focal-loss-in-pytorch-for-class-imbalance-24d8aa3b59d9"
    # Binary Cross-Entropy loss calculation
    bce_loss = F.cross_entropy(inputs, targets, reduction='none', **kwargs)
    pt = torch.exp(-bce_loss)  # Convert BCE loss to probability
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss  # Apply focal adjustment
    return focal_loss.mean()
