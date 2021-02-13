from typing import List, Tuple, Union

import cv2
import numpy as np
from sklearn.decomposition import PCA

from dimred.models import BaseModel


class PCAModel(BaseModel):
    def __init__(self, num_components: Union[int, float]):
        self.num_components = num_components

    def compress(self, img: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Compresses each image channel using PCA decomposition
        """
        result = list()
        for channel in cv2.split(img):
            num_components = self._get_num_components(channel)
            pca = PCA(n_components=num_components)
            compressed_channel = pca.fit_transform(channel)
            result.append(
                (compressed_channel.astype("float16"), pca.components_.astype("float16"), pca.mean_.astype("float16")))
        return result

    def decompress(self, embedding: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Decode PCA decomposition into original image
        """
        channels = [compressed_channel @ components + mean for compressed_channel, components, mean in embedding]
        return np.array(channels).transpose((1, 2, 0)).clip(0, 255).astype("uint8")

    def _get_num_components(self, channel):
        samples, features = channel.shape
        if type(self.num_components) is float:
            num_components = int(features * self.num_components)
        elif type(self.num_components) is int:
            num_components = self.num_components
        else:
            raise ValueError("num_components must be float or int")
        if num_components > features:
            raise ValueError("Number of components cannot be bigger than number of features")
        return num_components
