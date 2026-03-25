import os
import yaml


def create_directories():
    

    with open("./config.yaml", "r") as f:
        config = yaml.safe_load(f)

    for e in config["paths"]:
        pth = config["paths"][e]
        print("creating path ",e, ":", pth)
        os.makedirs(pth, exist_ok=True)


if __name__ == "__main__":
    create_directories()
