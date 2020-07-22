import torch
import torch.nn as nn
from catalyst.dl import registry


def patch_efficientnet_backbone(
    base_model: dict,
    checkpoint_model: dict,
    checkpoint: str
) -> nn.Module:
    """Initialize base model with weights from checkpoint model.

    Args:
        base_model (dict): base model initialization parameters
        checkpoint_model (dict): checkpoint model initialization parameters
        checkpoint (str): path to checkpoint (.pth file)

    Returns:
        nn.Module: base model with weights from checkpoint
    """
    base_model = registry.MODELS.get_from_params(**base_model)

    checkpoint_model = registry.MODELS.get_from_params(**checkpoint_model)
    state_dict = torch.load(checkpoint, map_location="cpu")["model_state_dict"]
    checkpoint_model.load_state_dict(state_dict)

    if hasattr(base_model, "backbone") and hasattr(checkpoint_model, "backbone"):
        if base_model.backbone._fc.in_features != checkpoint_model.backbone._fc.in_features:
            correct_fc = base_model.backbone._fc
            base_model.backbone = checkpoint_model.backbone
            base_model.backbone._fc = correct_fc
        else:
            base_model.backbone = checkpoint_model.backbone
    return base_model


def patch_efficientnet_conv_stem(
    base_model: dict,
    checkpoint_model: dict,
    checkpoint: str,
) -> nn.Module:
    """Initialize base model with weights from checkpoint model.

    Args:
        base_model (dict): base model initialization parameters
        checkpoint_model (dict): checkpoint model initialization parameters
        checkpoint (str): path to checkpoint (.pth file)

    Returns:
        nn.Module: base model with weights from checkpoint
    """
    base_model = registry.MODELS.get_from_params(**base_model)

    checkpoint_model = registry.MODELS.get_from_params(**checkpoint_model)
    state_dict = torch.load(checkpoint, map_location="cpu")["model_state_dict"]
    checkpoint_model.load_state_dict(state_dict)

    if hasattr(base_model, "backbone") and hasattr(checkpoint_model, "backbone"):
        correct_layer = base_model.backbone._conv_stem
        base_model.backbone = checkpoint_model.backbone
        base_model.backbone._conv_stem = correct_layer
    return base_model
