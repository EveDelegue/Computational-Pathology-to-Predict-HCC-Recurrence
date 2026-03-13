import argparse
import yaml
import os
from utils.utils import generate_patches_from_wsi


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide_name", type=str)
    args = parser.parse_args()
    return args


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    vis_scale = config["patching"]["vis_scale"]
    enlarge = config["patching"]["enlarge"]
    perc_wpx = config["patching"]["perc_wpx"]
    perc_bpx = config["patching"]["perc_bpx"]
    patches_path = config["paths"]["pth_to_patches"]
    features_path = config["paths"]["pth_to_features"]
    coords_path = config["paths"]["pth_to_coords"]
    overview_path = "results/overview_wsis"

    args = parse_arguments()
    sn = args.slide_name
    path_to_wsis = sn.split("Patient")[0]
    slide_name = "Patient_" + sn.split("_")[-1]
    if "PB" in sn:
        patch_size = step = config["patching"]["patch_size"]
    elif "HM" in sn:
        patch_size = step = config["patching"]["patch_size_hm"]
    else:
        patch_size = step = config["patching"]["patch_size_bj"]
    if slide_name.split("/")[-1].split(".")[0]+'_features.pt' not in os.listdir(features_path):
        generate_patches_from_wsi(
            slide_name,
            path_to_wsi=path_to_wsis,
            patch_size=patch_size,
            step=step,
            path_to_patches=patches_path,
            vis_scale=vis_scale,
            overview_path=overview_path,
            coords_path=coords_path,
            perc_wpx=perc_wpx,
            perc_bpx=perc_bpx,
            enlarge=enlarge,
            verbose=False,
        )
    else:
        print(slide_name.split("/")[-1].split(".")[0]+'_features.pt', "exists")


if __name__ == "__main__":
    main()
