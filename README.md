
## Kaggle: [ALASKA2 Image Steganalysis](https://www.kaggle.com/c/alaska2-image-steganalysis)


## Resizing data

```python
import os
import sys
import glob
# installed
import cv2
from tqdm import tqdm
from joblib import Parallel, delayed

N_JOBS = os.cpu_count() // 2
SIZES = (512, 512)


def _read_resize_save(from_: str, to_: str) -> None:
    """
    Read image to ram, resize and save it to file.
    Args:
        from_ (str): path to image to read
        to_ (str): path to file to save
    """
    img = cv2.imread(from_)
    resized = cv2.resize(img, SIZES, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(to_, resized)


def main(from_dir: str, to_dir: str) -> None:
    """
    Resize images from directory and save them to another.
    Args:
        from_dir (str): directory with .jpg images
        to_dir (str): directory to use for storing resized images
    """
    if not (os.path.exists(to_dir) and os.path.isdir(to_dir)):
        os.mkdir(to_dir)
    Parallel(N_JOBS)(
        delayed(_read_resize_save)(image, image.replace(from_dir, to_dir))
        for image in tqdm(glob.iglob(f"{from_dir}/*.jpg"))
    )


if __name__ == "__main__":
    from_dir, to_dir = sys.argv[1], sys.argv[2]
    main(from_dir, to_dir)

```