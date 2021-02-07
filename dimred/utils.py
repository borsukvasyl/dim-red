import os

import yaml


def get_relative_path(x: str, rel_to: str) -> str:
    return os.path.join(os.path.dirname(rel_to), x)


def load_yaml(x: str):
    with open(x) as fd:
        config = yaml.load(fd, yaml.FullLoader)
        config["yaml_path"] = x
        return config
