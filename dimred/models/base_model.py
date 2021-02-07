from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def compress(self):
        pass
