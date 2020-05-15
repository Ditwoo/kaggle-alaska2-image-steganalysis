import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class BinaryEfficientNet(nn.Module):
    def __init__(self, pretrain: str):
        """
        Args:
            pretrain (str): one of:
                - 'efficientnet-b0'
                - 'efficientnet-b1'
                - 'efficientnet-b2'
                - 'efficientnet-b3'
                - 'efficientnet-b4'
                - 'efficientnet-b5'
                - 'efficientnet-b6'
                - 'efficientnet-b7'
        """
        super().__init__()
        self.backbone = EfficientNet.from_pretrained(pretrain)
        self.backbone._fc = nn.Linear(self.backbone._fc.in_features, 1)
    
    def forward(self, batch):
        return self.backbone(batch)
