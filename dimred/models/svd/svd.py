from typing import List, Tuple, Union

import cv2
import numpy as np

from dimred.models import BaseModel


class SVDModel(BaseModel):
    def __init__(self, num_components: Union[int, float]):
        self.num_components = num_components

    def compress(self, img: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Compresses each image channel using SVD decomposition
        """
        result = []
        for channel in cv2.split(img):
            U, s, VT = np.linalg.svd(channel)
            num_components = self._get_num_components(s)
            U, s, VT = U[:, :num_components], s[:num_components], VT[:num_components, :]
            U, s, VT = U.astype(np.float16), s, VT.astype(np.float16)
            result.append((U, s, VT))
        return result

    def decompress(self, embedding: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Decode SVD decomposition into original image
        """
        channels = [U @ np.diag(s) @ VT for U, s, VT in embedding]
        img = np.dstack(channels)
        clipped_img = np.clip(img, 0, 255)
        return clipped_img.astype(np.uint8)

    def _get_num_components(self, s: np.ndarray) -> int:
        """
        Get number of components depending is it int or float
        """
        if type(self.num_components) is float:
            num_components = int(len(s) * self.num_components)
        elif type(self.num_components) is int:
            num_components = self.num_components
        else:
            raise ValueError("num_components must be float or int")
        if num_components > len(s):
            raise ValueError("Number of components cannot be bigger than number of features")
        return num_components
