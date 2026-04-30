import os
import yaml


def create_directories():
    

    with open("./config.yaml", "r") as f:
        config = yaml.safe_load(f)
    try:
        # give rights so the created content can be used
        og_um = os.umask(0)
        for e in config["paths"]:
            pth = config["paths"][e]
            print("creating path ",e, ":", pth)
            os.makedirs(pth, exist_ok=True)
    finally:
        # set umask back to it's original value
        os.umask(og_um)


if __name__ == "__main__":
    create_directories()
