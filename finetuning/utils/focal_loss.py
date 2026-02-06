import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

class BinaryFocalLoss(nn.Module):
    """Implementation of focal loss for binary classification"""
    def __init__(self, alpha: float = -1, gamma: float = 2.0, reduction: str = 'mean') -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        print("Setting focal loss", alpha, gamma, reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 2 and logits.size(1) == 2:
            logits = logits[:, 1]
        targets = targets.float()
        return sigmoid_focal_loss(logits, targets, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)