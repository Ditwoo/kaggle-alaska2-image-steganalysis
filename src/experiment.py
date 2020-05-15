import os
from collections import OrderedDict
# installed packages
import albumentations as alb
import albumentations.pytorch
from pandas import read_csv
from catalyst.dl import ConfigExperiment
# local files
from .datasets import ImagesDataset as Dataset
from .datasets import my_collator


TRAIN_AUGMENTATIONS = alb.Compose([
    alb.Resize(512, 512),
    alb.VerticalFlip(p=0.5),
    alb.HorizontalFlip(p=0.5),
    alb.JpegCompression(quality_lower=75, quality_upper=100, p=0.5),
    alb.Normalize(),
    # alb.ToFloat(max_value=255),
    alb.pytorch.ToTensorV2(),
])

VALID_AUGMENTATIONS = alb.Compose([
    alb.Resize(512, 512),
    alb.Normalize(),
    # alb.ToFloat(max_value=255),
    alb.pytorch.ToTensorV2(),
])


class Experiment(ConfigExperiment):

    def get_datasets(self, stage: str, folds: str, fold_index: int = None) -> OrderedDict:
        """

        Arguments:
            stage (str): stage name
            folds (str): path to csv with folds
            fold_index (str): fold index to use as validation set

        Returns:
            orderd dict with train & valid datasets
        """

        fold_index = os.environ.get("FOLD_INDEX") or fold_index
        if fold_index is None:
            raise ValueError("Should be specified 'fold_index'!")

        datasets = OrderedDict()

        df = read_csv(folds)
        train = df[df["folds"] != fold_index]

        datasets["train"] = {
            "dataset": Dataset(
                images=train["images"].values, 
                labels=train["labels"].values,
                transforms=TRAIN_AUGMENTATIONS,
            ),
            "shuffle": True,
            # "collate_fn": my_collator,
        }
        print(f" * Num records in train dataset: {train.shape[0]}", flush=True)

        valid = df[df["folds"] == fold_index]
        datasets["valid"] = {
            "dataset": Dataset(
                images=valid["images"].values, 
                labels=valid["labels"].values,
                transforms=VALID_AUGMENTATIONS,
            ),
            "shuffle": False,
            # "collate_fn": my_collator,
        }
        print(f" * Num records in valid dataset: {valid.shape[0]}", flush=True)

        return datasets
