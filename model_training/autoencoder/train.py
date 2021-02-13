from torch.utils.data import DataLoader
import torch as T
from pathlib import Path
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from dimred.models.autoencoder.config.auto_encoder_cfg import config
from dimred.models.autoencoder.data_loader import ImageDataset
from dimred.models.autoencoder.autoencoder import AutoEncoderModel

import numpy as np
from fire import Fire


def main(images_path: str, model_path: str):
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

        print(f'epoch {epoch+1}/{config.model_config.epochs}, loss:{loss.item()}')
        T.save(autoencoder.state_dict(), Path(model_path) / f"checkpoint_model_{epoch}.pth")
    T.save(autoencoder.state_dict(), Path(model_path) / "autoencoder.pth")


if __name__ == "__main__":
    Fire(main)
