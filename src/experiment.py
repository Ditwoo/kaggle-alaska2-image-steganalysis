import os
from functools import partial
from collections import OrderedDict
# installed packages
import numpy as np
import albumentations as alb
import albumentations.pytorch
from pandas import read_csv
import torch
import torch.nn as nn
from torch.utils.data.sampler import WeightedRandomSampler
from catalyst.dl import ConfigExperiment
from catalyst.data.sampler import BalanceClassSampler
# local files
from .datasets import (
    ImagesDataset,
    OneHotLabelsImagesDataset
)


def _drop_center(image, **kwargs):
    image[128:256+128, 128:256+128, :] = 0
    return image


def _block_coarse_dropout(image, **kwargs):
    num_blocks_to_drop_min = kwargs.get('num_blocks_to_drop_min', 40)
    num_blocks_to_drop_max = kwargs.get('num_blocks_to_drop_max', 100)
    num_blocks_to_drop = np.random.randint(
        low=num_blocks_to_drop_min, high=num_blocks_to_drop_max
    )
    num_blocks_x = 512 / 8
    num_blocks_y = 512 / 8
    coordinate_blocks_to_drop_x = np.random.randint(
        low=0, high=num_blocks_x, size=num_blocks_to_drop
    )
    coordinate_blocks_to_drop_y = np.random.randint(
        low=0, high=num_blocks_y, size=num_blocks_to_drop
    )
    for i in range(num_blocks_to_drop):
        image[
            coordinate_blocks_to_drop_x[i]*8 : coordinate_blocks_to_drop_x[i]*8+8,
            coordinate_blocks_to_drop_y[i]*8 : coordinate_blocks_to_drop_y[i]*8+8,
            :
        ] = 0
    return image


TRAIN_AUGMENTATIONS = alb.Compose([
    # alb.Resize(512, 512),
    alb.VerticalFlip(p=0.5),
    alb.HorizontalFlip(p=0.5),

    alb.OneOf(
        [
            alb.Lambda(image=partial(_drop_center), p=1.0),
            alb.Lambda(image=partial(_block_coarse_dropout), p=1.0)
        ], p=0.6
    ),

    alb.Normalize(),
    alb.pytorch.ToTensorV2(),
])

VALID_AUGMENTATIONS = alb.Compose([
    # alb.Resize(512, 512),
    alb.Normalize(),
    alb.pytorch.ToTensorV2(),
])


class Experiment(ConfigExperiment):

    def _postprocess_model_for_stage(self, stage: str, model: nn.Module) -> nn.Module:
        model_ = model
        if isinstance(model, torch.nn.DataParallel):
            model_ = model_.module
        if hasattr(model_, "backbone"):
            if stage.endswith("_backbone_tuning"):
                cnt_unfreezed = 0
                prefix = "_fc."
                for name, parameter in model_.backbone.named_parameters():
                    parameter.requires_grad = name.startswith(prefix)
                    cnt_unfreezed += name.startswith(prefix)
                print(f" * Unfreezed {cnt_unfreezed} parameters")
            elif stage.endswith("_conv_stem_tuning"):
                cnt_unfreezed = 0
                prefix = "_conv_stem."
                for name, parameter in model_.backbone.named_parameters():
                    parameter.requires_grad = name.startswith(prefix)
                    cnt_unfreezed += name.startswith(prefix)
                print(f" * Unfreezed {cnt_unfreezed} parameters")
            else:
                for param in model_.parameters():
                    param.requires_grad = True
                print(" * Nothing to freeze")
        return model_

    def get_datasets(self,
                     stage: str,
                     folds: str,
                     fold_index: int = None,
                     is_multiclass: bool = False,
                     use_sampling: bool = True,
                     bgr2rgb: bool = True) -> OrderedDict:
        """

        Arguments:
            stage (str): stage name
            folds (str): path to csv with folds
            fold_index (str): fold index to use as validation set
            is_multiclass (bool): train as multiclass, default False
            use_sampling (bool): use class sampling, default True

        Returns:
            orderd dict with train & valid datasets
        """

        fold_index = os.environ.get("FOLD_INDEX") or fold_index
        if fold_index is None:
            raise ValueError("Should be specified 'fold_index' or env variable 'FOLD_INDEX'!")

        Dataset = OneHotLabelsImagesDataset if is_multiclass else ImagesDataset

        datasets = OrderedDict()

        df = read_csv(folds)
        train = df[df["folds"] != fold_index]
        _images = train["images"].values
        _target = train["labels" if not is_multiclass else "classes"].values
        datasets["train"] = {
            "dataset": Dataset(
                images=_images,
                labels=_target,
                transforms=TRAIN_AUGMENTATIONS,
                bgr2rgb=bgr2rgb,
            ),
        }
        if use_sampling:
            datasets["train"]["sampler"] = BalanceClassSampler(
                labels=_target,
                mode="downsampling",
            )
            datasets["train"]["shuffle"] = False
        else:
            datasets["train"]["shuffle"] = True
        print(f" * Num records in train dataset: {train.shape[0]}", flush=True)

        valid = df[df["folds"] == fold_index]
        _images = valid["images"].values
        _target = valid["labels" if not is_multiclass else "classes"].values
        datasets["valid"] = {
            "dataset": Dataset(
                images=_images,
                labels=_target,
                transforms=VALID_AUGMENTATIONS,
                bgr2rgb=bgr2rgb,
            ),
            "shuffle": False,
        }
        print(f" * Num records in valid dataset: {valid.shape[0]}", flush=True)

        return datasets
