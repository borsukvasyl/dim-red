from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from dimred.utils import load_yaml, get_relative_path


class BaseModel(ABC):
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
