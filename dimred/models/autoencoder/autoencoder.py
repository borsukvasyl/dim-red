import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize

from dimred.models import BaseModel


class AutoEncoderModel(BaseModel, nn.Module):
    def __init__(self, model_path=None):
        super().__init__()
        self.encoded = None
        self._model()
        if model_path:
            self.load_state_dict(torch.load(model_path))
            self.eval()

    def _model(self):
        # ENCODER
        # output shape 64x64x64
        self.e_conv_1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(2, 2)
            ),
            nn.LeakyReLU(),
        )
        # output shape  128x32x32
        self.e_conv_2 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2)
            ),
            nn.LeakyReLU(),
        )
        # output shape  128x32x32
        self.e_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
        )
        # output shape  128x32x32
        self.e_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
        )

        # output shape  128x32x32
        self.e_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
        )

        # output shape  32x32x32
        self.e_conv_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=32,
                kernel_size=(5, 5),
                stride=(1, 1),
                padding=(2, 2),
            ),
            nn.Tanh(),
        )

        # DECODER

        # output shape  128x64x64
        self.d_up_conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2)
            ),
        )

        # output shape 28x64x64
        self.d_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
        )

        # output shape  128x64x64
        self.d_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
        )

        # output shape  128x64x64
        self.d_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
        )

        # output shape  256x128x128
        self.d_up_conv_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(
                in_channels=32, out_channels=256, kernel_size=(2, 2), stride=(2, 2)
            ),
        )

        # output shape  3x128x128
        self.d_up_conv_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=16, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ReflectionPad2d((2, 2, 2, 2)),
            nn.Conv2d(
                in_channels=16, out_channels=3, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.Tanh(),
        )

    def _encode(self, img):
        """
        Compresses each image Auto Encoder decomposition
        """
        ec1 = self.e_conv_1(img)
        ec2 = self.e_conv_2(ec1)
        eblock1 = self.e_block_1(ec2) + ec2
        eblock2 = self.e_block_2(eblock1) + eblock1
        eblock3 = self.e_block_3(eblock2) + eblock2
        ec3 = self.e_conv_3(eblock3)  # in [-1, 1] from tanh activation

        # encoded tensor
        self.encoded = 0.5 * (ec3 + 1)  # (-1|1) -> (0|1)
        return self.encoded

    def _decode(self, embedding):
        """
        Decode Auto Encoder decomposition into original image
        """
        y = embedding * 2.0 - 1  # transform from zero-one range to minus one - one

        uc1 = self.d_up_conv_1(y)
        dblock1 = self.d_block_1(uc1) + uc1
        dblock2 = self.d_block_2(dblock1) + dblock1
        dblock3 = self.d_block_3(dblock2) + dblock2
        uc2 = self.d_up_conv_2(dblock3)
        dec = self.d_up_conv_3(uc2)

        return dec

    def forward(self, img):
        embedding = self._encode(img)
        return self._decode(embedding)

    def compress(self, img: np.ndarray) -> np.ndarray:
        """
        Take single image as input and return embedding
        """
        img = resize(img, (400, 400), anti_aliasing=True) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = img.reshape((1, *img.shape))
        img = torch.from_numpy(img).float()
        return self._encode(img)

    def decompress(self, embedding: np.ndarray) -> np.ndarray:
        """
        Take embedding vector as input and return reconstructed image
        """
        decoded = self._decode(embedding)
        return decoded[0] * 255
