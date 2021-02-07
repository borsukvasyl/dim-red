from typing import Dict

import numpy as np
from pympler import asizeof
from sewar.full_ref import mse, ssim, vifp, psnrb, psnr

from dimred.models import BaseModel


def get_metrics(compression_model: BaseModel, original_image: np.ndarray) -> Dict[str, float]:
    compressed_image = compression_model.compress(original_image)
    decompressed_image = compression_model.decompress(compressed_image)
    return {
        "mse": mse(original_image, decompressed_image),
        "ssim": ssim(original_image, decompressed_image)[0],
        "vif-p": vifp(original_image, decompressed_image),
        "psnr-b": psnrb(original_image, decompressed_image),
        "psnr": psnr(original_image, decompressed_image),
        "original_image_size": asizeof.asizeof(original_image),
        "compressed_image_size": asizeof.asizeof(compressed_image),
        "compression_ratio":  asizeof.asizeof(original_image) / asizeof.asizeof(compressed_image)
    }
