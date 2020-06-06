from typing import Union, Tuple
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


class ImagesDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Union[tuple, torch.FloatTensor]:
        file = self.images[index]
        image = cv2.imread(file) # [:, :, -1]  # bgr -> rgb
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        if self.labels is None:
            return image
        label = torch.tensor([self.labels[index]], dtype=torch.float)
        return image, label


def _one_hot(size: int, target: int) -> torch.Tensor:
    out = torch.zeros(size, dtype=torch.float32)
    out[target] = 1.0
    return out


class OneHotLabelsImagesDataset(ImagesDataset):
    def __init__(self, images, labels=None, transforms=None):
        super().__init__(images, labels, transforms)
        self.n_classes = None if labels is None else (np.max(labels) + 1)

    def __getitem__(self, index: int):
        file = self.images[index]
        image = cv2.imread(file) # [:, :, -1]  # bgr -> rgb
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        if self.labels is None:
            return image
        label = _one_hot(self.n_classes, self.labels[index])
        return image, label


class MixupImagesDataset(ImagesDataset):
    def __init__(self,
                 images,
                 labels=None,
                 transforms=None,
                 mixup_alpha: Union[float, Tuple[float, float]] = 0.0,
                 mixup_prob: float = 0.0):
        super().__init__(images, labels, transforms)
        self.alpha = mixup_alpha
        self.mixup_prob = mixup_prob

    def __getitem__(self, index: int) -> Union[tuple, torch.FloatTensor]:
        file = self.images[index]
        image = cv2.imread(file) / 255.
        # sample alpha
        if isinstance(self.alpha, (tuple, list)):
            alpha = np.random.uniform(self.alpha[0], self.alpha[1])
        else:
            alpha = self.alpha
        # sample random image & do mixup
        if np.random.uniform(0, 1) < self.mixup_prob:
            random_index = np.random.randint(0, len(self.images))
            random_image = cv2.imread(self.images[random_index]) / 255.
            image = random_image * alpha + image * (1 - alpha)
        else:
            random_index = None
        # augment
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        if self.labels is None:
            return image
        # do mixup for label too
        label = self.labels[index]
        if random_index is not None:
            random_label = self.labels[random_index]
            label = random_label * alpha + label * (1 - alpha)

        label = torch.tensor([label], dtype=torch.float)
        return image, label


def my_collator(batch):
    xs = torch.stack([x for x, _ in batch])
    ys = torch.stack([y for _, y in batch]).view(-1)
    return xs, ys
