import glob

import joblib
import numpy as np
from fire import Fire
from skimage.io import imread
from sklearn.cluster import KMeans

from dimred.models.kmeans.kmeans import patch_image


def get_data(images, window_size: int):
    data = []
    for img in images:
        patches, _, _ = patch_image(img, window_size=window_size)
        data.append(patches)
    return np.vstack(data)


def fit_kmeans(images, window_size: int = 8, n_clusters: int = 2000):
    data = get_data(images, window_size=window_size)
    model = KMeans(n_clusters=n_clusters).fit(data)
    return model


def main(images_path: str, model_name: str, window_size: int = 8, n_clusters: int = 2000):
    images = [imread(path) for path in glob.glob(images_path)]
    model = fit_kmeans(images, window_size=window_size, n_clusters=n_clusters)
    joblib.dump(model, f"{model_name}.joblib")


if __name__ == "__main__":
    Fire(main)
