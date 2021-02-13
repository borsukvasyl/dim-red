from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from skimage.io import imread
from skimage.transform import resize
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root: str):
        self.files = sorted(Path(root).iterdir())
        self.files = [file for file in self.files if file.suffix == '.jpg']

    def __getitem__(self, idx: int) -> Tuple[T.Tensor, str]:
        path = str(self.files[idx % len(self.files)])

        img = imread(path)
        img = resize(img, (400, 400), anti_aliasing=True)/ 255.0

        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()

        return img, path

    def __len__(self):
        return len(self.files)
