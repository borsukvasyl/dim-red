from dimred.models import BaseModel


class AutoEncoderModel(BaseModel):
    def compress(self, img):
        pass

    def decompress(self, embedding):
        pass
