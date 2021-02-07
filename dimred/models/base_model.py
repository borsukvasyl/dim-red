from abc import ABC, abstractmethod
from typing import Optional

from dimred.utils import load_yaml, get_relative_path


class BaseModel(ABC):
    def get_metrics(self, img):
        embedding = self.compress(img)
        restored = self.decompress(embedding)
        return {"restored": restored}

    @abstractmethod
    def compress(self, img):
        pass

    @abstractmethod
    def decompress(self, embedding):
        pass

    @classmethod
    def from_config(cls, config_path: Optional[str] = None):
        if config_path is not None:
            config_path = get_relative_path(config_path, __file__)
            config = load_yaml(config_path)
            return cls(**config)
        return cls()
