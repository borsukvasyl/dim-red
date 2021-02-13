from easydict import EasyDict
import torch as T

config = EasyDict()
config.device = "cuda" if T.cuda.is_available() else "cpu"
config.model_config = EasyDict()
config.model_config.epochs = 10
config.model_config.batch_size = 128
config.model_config.weight_decay = 1e-5
config.model_config.learning_rate = 10e-4
