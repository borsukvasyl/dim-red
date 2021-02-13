from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from easydict import EasyDict
from fire import Fire
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dimred.models.autoencoder.autoencoder import AutoEncoderModel
from model_training.autoencoder.data_loader import ImageDataset


def get_config():
    config = EasyDict()
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.model_config = EasyDict()
    config.model_config.epochs = 10
    config.model_config.batch_size = 128
    config.model_config.weight_decay = 1e-5
    config.model_config.learning_rate = 10e-4
    return config


def main(images_path: str, model_path: str):
    config = get_config()
    dataloader = DataLoader(
        dataset=ImageDataset(images_path),
        batch_size=config.model_config.batch_size,
    )

    if config.device == "cuda":
        autoencoder = AutoEncoderModel().cuda()
    else:
        autoencoder = AutoEncoderModel().cpu()
    optimizer = optim.Adam(autoencoder.parameters(), lr=config.model_config.learning_rate,
                           weight_decay=config.model_config.weight_decay)
    loss_criterion = nn.MSELoss()
    for epoch in range(config.model_config.epochs):
        for data in dataloader:
            img, _ = data
            img = Variable(img).cpu()

            output = autoencoder(img)
            loss = loss_criterion(output, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"epoch {epoch + 1}/{config.model_config.epochs}, loss:{loss.item()}")
        torch.save(autoencoder.state_dict(), Path(model_path) / f"checkpoint_model_{epoch}.pth")

    torch.save(autoencoder.state_dict(), Path(model_path) / "autoencoder.pth")


if __name__ == "__main__":
    Fire(main)
