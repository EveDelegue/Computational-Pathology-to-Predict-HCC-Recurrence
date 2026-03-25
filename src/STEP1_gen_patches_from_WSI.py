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
    coords_path = config["paths"]["pth_to_coords"]
    overview_path = config["paths"]["overview_path"]

    args = parse_arguments()
    sn = args.slide_name
    path_to_wsis = sn.split("Patient")[0]
    hospital_name = sn.split(os.path.sep)[-3]
    slide_name = "Patient_" + sn.split("_")[-1]

    patch_size_dict = config["patching"]["patch_size_dict"]
    if hospital_name not in patch_size_dict.keys():
        
        raise KeyError(f"no resolution defined for this hospital : {hospital_name}. Check that the hospital's name in the config file is the same as in the data folder.")

    patch_size = step = patch_size_dict[hospital_name]

    if slide_name.split("/")[-1].split(".")[0] not in os.listdir(patches_path):
        generate_patches_from_wsi(
            slide_name,
            path_to_wsi=path_to_wsis,
            patch_size=patch_size,
            step=step,
            path_to_patches=patches_path,
            vis_scale=vis_scale,
            overview_path=overview_path,
            hospital_name=hospital_name,
            coords_path=coords_path,
            perc_wpx=perc_wpx,
            perc_bpx=perc_bpx,
            enlarge=enlarge,
        )
    else:
        print(slide_name, "exists")


if __name__ == "__main__":
    main()
