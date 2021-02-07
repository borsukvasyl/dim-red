from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import numpy as np
from sewar.full_ref import mse, ssim, vifp, msssim

from dimred.utils import load_yaml, get_relative_path


class BaseModel(ABC):
    def get_metrics(self, original_image: np.ndarray, compressed_image: Optional[np.ndarray] = None)\
            -> Dict[str, float]:
        if compressed_image is None:
            compressed_image = self.decompress(self.compress(original_image))
        return {
            "mse": mse(original_image, compressed_image),
            "ssim": ssim(original_image, compressed_image),
            "vif-p": vifp(original_image, compressed_image),
            "ms-ssim": msssim(original_image, compressed_image)
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
