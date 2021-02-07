from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import numpy as np
from pympler import asizeof
from sewar.full_ref import mse, ssim, vifp, psnrb, psnr

from dimred.utils import load_yaml, get_relative_path


class BaseModel(ABC):
    def get_metrics(self, original_image: np.ndarray) -> Dict[str, float]:
        compressed_image = self.compress(original_image)
        decompressed_image = self.decompress(compressed_image)
        return {
            "mse": mse(original_image, decompressed_image),
            "ssim": ssim(original_image, decompressed_image)[0],
            "vif-p": vifp(original_image, decompressed_image),
            "psnr-b": psnrb(original_image, decompressed_image),
            "psnr": psnr(original_image, decompressed_image),
            "saved_memory_ratio": asizeof.asizeof(compressed_image) / asizeof.asizeof(original_image)
        }

    @abstractmethod
    def compress(self, img: np.ndarray) -> Any:
        pass

    @abstractmethod
    def decompress(self, embedding: Any) -> np.ndarray:
        pass

    @classmethod
    def from_config(cls, config_path: Optional[str] = None) -> "BaseModel":
        if config_path is not None:
            config_path = get_relative_path(config_path, __file__)
            config = load_yaml(config_path)
            return cls(**config)
        return cls()
