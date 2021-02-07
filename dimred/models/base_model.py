from abc import ABC, abstractmethod


class BaseModel(ABC):
    def get_metrics(self, img):
        embedding = self.compress(img)
        restored = self.decompress(embedding)
        return {"koko": True}

    @abstractmethod
    def compress(self, img):
        pass

    @abstractmethod
    def decompress(self, embedding):
        pass
