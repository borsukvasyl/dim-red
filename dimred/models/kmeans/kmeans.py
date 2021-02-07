import numpy as np
from patchify import patchify, unpatchify
from sklearn.cluster import KMeans

from dimred.models import BaseModel


def calculate_padding(img: np.ndarray, window: int):
    h, w = img.shape[0], img.shape[1]
    padding_fn = lambda x: 0 if x % window == 0 else window - x % window
    padding_h = padding_fn(h)
    padding_w = padding_fn(w)
    return padding_h, padding_w


def patch_image(img: np.ndarray, window_size: int):
    padding_h, padding_w = calculate_padding(img, window_size)
    img_pad = np.pad(img, ((0, padding_h), (0, padding_w), (0, 0)))
    patches = patchify(img_pad, (window_size, window_size, 3), step=window_size)
    patches = np.transpose(patches, (0, 1, 2, 5, 3, 4))
    patches = patches.reshape(-1, window_size * window_size)
    return patches, padding_h, padding_w


def unpatch_image(patches: np.ndarray, window_size: int, padding_h: int, padding_w: int, img_shape: tuple):
    h, w = img_shape[0] + padding_h, img_shape[1] + padding_w
    rows, cols = h // window_size, w // window_size
    patches = patches.reshape(rows, cols, 1, 3, window_size, window_size)
    patches = np.transpose(patches, (0, 1, 2, 4, 5, 3))
    img = unpatchify(patches, (h, w, 3))
    return img


class KMeansModel(BaseModel):
    def __init__(self, window_size: int, train_img: np.ndarray, n_clusters: int = 1000):
        self.window_size = window_size

        patches, padding_h, padding_w = patch_image(train_img, self.window_size)
        self.kmeans = KMeans(n_clusters=n_clusters).fit(patches)

    def compress(self, img):
        patches, padding_h, padding_w = patch_image(img, self.window_size)
        embedding = self.kmeans.predict(patches)
        return embedding, padding_h, padding_w, img.shape

    def decompress(self, embedding):
        embedding, padding_h, padding_w, img_shape = embedding
        clusters = self.kmeans.cluster_centers_
        patches = clusters[embedding]
        img = unpatch_image(patches, self.window_size, padding_h, padding_w, img_shape)
        return img
