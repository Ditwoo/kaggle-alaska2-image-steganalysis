import os
from collections import OrderedDict
# installed packages
import albumentations as alb
import albumentations.pytorch
from pandas import read_csv
from torch.utils.data.sampler import WeightedRandomSampler
from catalyst.dl import ConfigExperiment
from catalyst.data.sampler import BalanceClassSampler
# local files
from .datasets import (
    ImagesDataset,
    OneHotLabelsImagesDataset
)


TRAIN_AUGMENTATIONS = alb.Compose([
    # alb.Resize(512, 512),
    alb.VerticalFlip(p=0.5),
    alb.HorizontalFlip(p=0.5),
    alb.Normalize(),
    alb.pytorch.ToTensorV2(),
])

VALID_AUGMENTATIONS = alb.Compose([
    # alb.Resize(512, 512),
    alb.Normalize(),
    alb.pytorch.ToTensorV2(),
])


# TODO: label smoothing criterion

class Experiment(ConfigExperiment):

    def get_datasets(self,
                     stage: str,
                     folds: str,
                     fold_index: int = None,
                     is_multiclass: bool = False,
                     use_sampling: bool = True) -> OrderedDict:
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
            ),
            "shuffle": True,
        }
        print(f" * Num records in valid dataset: {valid.shape[0]}", flush=True)

        return datasets
