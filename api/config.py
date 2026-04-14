import os
import yaml

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, "../config.yml")


def get_full_path(rel_path: str) -> str:
    return os.path.normpath(os.path.join(ROOT_DIR, "..", rel_path))


with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.load(f, Loader=yaml.SafeLoader)
    for key, value in CONFIG["paths"].items():
        CONFIG["paths"][key] = get_full_path(value)
