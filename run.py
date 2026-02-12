from utils.io import load_config
from core.train import train

if __name__ == "__main__":
    config = load_config("config.yaml")
    train(config)
