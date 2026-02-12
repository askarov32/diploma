from core.train import train
from apputils.io import load_config

if __name__ == "__main__":
    config = load_config("config_cut.yaml")
    print(config)
    train(config)
