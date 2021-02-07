from typing import List, Tuple

import cv2
import numpy as np

from dimred.models import BaseModel


class SVDModel(BaseModel):
    def __init__(self, num_components: int):
        self.num_components = num_components

    def compress(self, img: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Compresses each image channel using SVD decomposition
        """
        result = []
        for channel in cv2.split(img):
            U, s, VT = np.linalg.svd(channel)
            U, s, VT = U[:, :self.num_components], s[:self.num_components], VT[:self.num_components, :]
            result.append((U, s, VT))
        return result

    def decompress(self, embedding: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Decode SVD decomposition into original image
        """
        channels = [U @ np.diag(s) @ VT for U, s, VT in embedding]
        img = cv2.merge(channels)
        clipped_img = np.clip(img, 0, 255)
        return clipped_img.astype(np.uint8)
