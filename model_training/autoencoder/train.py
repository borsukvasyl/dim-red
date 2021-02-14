import glob
from typing import Tuple
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from easydict import EasyDict
from fire import Fire
from skimage.io import imread
from skimage.transform import resize
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from dimred.models.autoencoder.autoencoder import AutoEncoderModel


class ImageDataset(Dataset):
    def __init__(self, root: str):
        self.files = sorted(glob.glob(os.path.join(root, "*.jpg")))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.files[idx % len(self.files)]
        img = imread(path)
        img = resize(img, (224, 224), anti_aliasing=True)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        return img

    def __len__(self):
        return len(self.files)


def get_config():
    config = EasyDict()
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.model_config = EasyDict()
    config.model_config.epochs = 200
    config.model_config.batch_size = 24
    config.model_config.weight_decay = 0.01
    config.model_config.learning_rate = 0.0001
    return config


def main(images_path: str, model_name: str):
    config = get_config()
    dataloader = DataLoader(
        dataset=ImageDataset(images_path),
        batch_size=config.model_config.batch_size,
    )
    autoencoder = AutoEncoderModel()
    autoencoder.cuda().eval()
    optimizer = optim.AdamW(autoencoder.parameters(), lr=config.model_config.learning_rate,
                            weight_decay=config.model_config.weight_decay)
    loss_criterion = nn.BCELoss()
    pbar = tqdm(total=config.model_config.epochs)
    for epoch in range(config.model_config.epochs):
        for img in dataloader:
            optimizer.zero_grad()
            img = img.cuda() if config.device == "cuda" else img.cpu()
            output = autoencoder(img)
            loss = loss_criterion(output, img)
            loss.backward()
            optimizer.step()

        pbar.set_description(f"epoch {epoch + 1}/{config.model_config.epochs}, loss:{loss.item()}")
        pbar.update()
    torch.save(autoencoder.state_dict(), f"{model_name}.pth")


if __name__ == "__main__":
    Fire(main)
