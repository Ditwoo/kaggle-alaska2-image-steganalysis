import torch.nn as nn
from torchvision.models import densenet


_backbones = {
    "densenet121": densenet.densenet121,
    "densenet161": densenet.densenet161,
    "densenet169": densenet.densenet169,
    "densenet201": densenet.densenet201,
}


class BinaryDensenet(nn.Module):
    def __init__(self, pretrain: str):
        """
        Args: 
            backbone (str): should be one of:
                - 'densenet121'
                - 'densenet161'
                - 'densenet169'
                - 'densenet201'
        """
        super().__init__()
        self.backbone = _backbones[pretrain](pretrained=True)
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, 1)

    def forward(self, batch):
        return self.backbone(batch)
