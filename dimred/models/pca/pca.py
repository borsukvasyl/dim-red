from typing import List, Tuple

import cv2
import numpy as np
from sklearn.decomposition import PCA as _PCA

from dimred.models import BaseModel


class PCA(BaseModel):
    def __init__(self, num_components: int):
        self.num_components = num_components

    def compress(self, img: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Compresses each image channel using PCA decomposition
        """
        result = list()
        for channel in cv2.split(img):
            pca = _PCA(n_components=self.num_components)
            compressed_channel = pca.fit_transform(channel)
            result.append((compressed_channel, pca.components_, pca.mean_))
        return result

    def decompress(self, embedding: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Decode PCA decomposition into original image
        """
        channels = [compressed_channel @ components + mean for compressed_channel, components, mean in embedding]
        img = cv2.merge(channels)
        clipped_img = np.clip(img, 0, 255)
        return clipped_img.astype(np.uint8)
