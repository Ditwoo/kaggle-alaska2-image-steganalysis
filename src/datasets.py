from typing import Union, Tuple
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


def my_collator(batch):
    xs = torch.stack([x for x, _ in batch])
    ys = torch.stack([y for _, y in batch]).view(-1)
    return xs, ys
