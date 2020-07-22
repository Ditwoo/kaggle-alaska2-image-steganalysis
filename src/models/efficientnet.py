import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import Conv2dStaticSamePadding


class MulticlassEfficientNet(nn.Module):
    def __init__(self, pretrain: str, n_classes: int = 1):
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
        self.backbone._fc = nn.Linear(self.backbone._fc.in_features, n_classes)

    def forward(self, batch):
        return self.backbone(batch)


class StemMulticlassEfficientNet(MulticlassEfficientNet):
    def __init__(self, pretrain: str, n_classes: int = 1):
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
        super().__init__(pretrain, n_classes)
        self.backbone._conv_stem = Conv2dStaticSamePadding(
            self.backbone._conv_stem.in_channels,
            self.backbone._conv_stem.out_channels,
            kernel_size=3,
            stride=2,  # lower stride ?
            image_size=224,
            dilation=2,
            bias=False
        )


class BinaryEfficientNet(MulticlassEfficientNet):
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
        super().__init__(pretrain, 1)


# TODO: initialization from other backbones
class LLFEfficientNet(nn.Module):
    """EfficientNet with additional low
    level features added to head.
    """

    def __init__(self, pretrain: str, n_classes: int = 1):
        super().__init__()

        self.backbone = EfficientNet.from_pretrained(pretrain)
        low_level_features = 40 + 448
        features = self.backbone._fc.in_features + low_level_features
        self.backbone._fc = nn.Linear(features, n_classes)
    
    def extract_features(self, inputs):
        """use convolution layer to extract feature .
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

            if idx == len(self.backbone._blocks) // 2:
                mid_x = self.backbone._swish(x)

            if idx == len(self.backbone._blocks) // 4:
                quarter_x = self.backbone._swish(x)

        # Head
        x = self.backbone._swish(self.backbone._bn1(self.backbone._conv_head(x)))

        return x, mid_x, quarter_x

    def forward(self, inputs):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of this model after processing.
        """
        bs = inputs.size(0)

        # Convolution layers
        x, mid_x, quarter_x = self.extract_features(inputs)
        
        mid_x = F.adaptive_avg_pool2d(mid_x, 2)
        mid_x = mid_x.view(bs, -1)
        
        quarter_x = F.adaptive_avg_pool2d(quarter_x, 1)
        quarter_x = quarter_x.view(bs, -1)

        # Pooling and final linear layer
        x = self.backbone._avg_pooling(x)
        x = x.view(bs, -1)

        x = torch.cat([x, mid_x, quarter_x], 1)

        x = self.backbone._dropout(x)
        x = self.backbone._fc(x)

        return x
