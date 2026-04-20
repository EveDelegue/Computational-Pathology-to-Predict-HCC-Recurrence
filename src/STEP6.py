import torch
import cv2
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  PIL import Image
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import argparse
from utils.utils_tumor import (
    gen_image_from_coords,
    gen_image_from_coords_bis,
)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose",type=bool,default=True)
    args = parser.parse_args()
    return args

def main():
    # read hyperparameters
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # parse arguments
    args = parse_arguments()
    verbose = args.verbose

    patch_size = config["patching"]["patch_size_dict"]["PB"]
    tia_patch_size = config["patching"]["tia_patch_size"]

    colors = {
        0: config["visualization"]["colors"]["healthy"],
        1: config["visualization"]["colors"]["non_pej"],
        2: config["visualization"]["colors"]["pej"],
    }
    colors_TNT = {
        0: config["visualization"]["colors"]["healthy"],
        1: config["visualization"]["colors"]["pej"],
        2: config["visualization"]["colors"]["pej"],
    }

    vis_scale = config["patching"]["vis_scale"]

    # TODO : adapt the step for inflammation 
    step = int(vis_scale * patch_size)
    padding = 2 * step

    step_inflam = int(vis_scale * tia_patch_size)
    padding_inflam = 2 * step_inflam

    tumor_checkpoints =  config["paths"]["pth_to_tumor_ckpts"]
    coords_checkpoints =  config["paths"]["pth_to_coords"]
    inflams_checkpoints =  config["paths"]["pth_to_inflams_ckpts"]
    inflams_verbose = config["paths"]["pth_to_inflams_wsis"]


    # create a dataframe
    cols = ["lame", "peri-tumoral", "intra-tumoral"]
    df = pd.DataFrame(columns=cols)
    # fill the "lame" column
    df["lame"] = [
        slide_name.split("_")[0] for slide_name in os.listdir(inflams_checkpoints)
    ]
    slides = os.listdir(inflams_checkpoints)

    # for each lame,
    for slide in slides:
        slide_name = slide.split("_")[0]
        hospital = slide.split("_")[1]
        # load the tumoral predictions
        chkpt_coords = torch.load(f"{coords_checkpoints}/{slide_name}_{hospital}_coords_checkpoint.pt",weights_only=False)
        chkpt_tumor = torch.load(
            f"{tumor_checkpoints}/{slide_name}_{hospital}_preds_probas_checkpoint.pt",weights_only=False
        )
        chkpt_inflam = torch.load(
            f"{inflams_checkpoints}/{slide_name}_{hospital}_coords_inflams_checkpoint.pt",weights_only=False
        )

        [x_start, y_start, _, _] = chkpt_coords["xy_start_end"]

        # new pd frame for tumor
        df_tumor = pd.DataFrame()
        # copy from the loaded ones
        df_tumor["x"] = chkpt_tumor["coords_x"]
        df_tumor["y"] = chkpt_tumor["coords_y"]
        df_tumor["tumor"] = [p.item() for p in chkpt_tumor["har_mean_preds"]]
        df_tumor["xx"] = df_tumor["x"] * vis_scale - x_start
        df_tumor["yy"] = df_tumor["y"] * vis_scale - y_start

        coords_x_tum = df_tumor["xx"].values
        coords_y_tum = df_tumor["yy"].values
        preds = df_tumor["tumor"].values

        # new pd frame for inflam
        df_inflams = pd.DataFrame()
        df_inflams["x"] = chkpt_inflam["coords_x"]
        df_inflams["y"] = chkpt_inflam["coords_y"]
        df_inflams["inflams"] = chkpt_inflam["inflams"]
        
        # downscale slide coordinates
        df_inflams["xx"] = df_inflams["x"]* vis_scale - x_start
        df_inflams["yy"] = df_inflams["y"]* vis_scale - y_start

        coords_x_inf = df_inflams["xx"].values
        coords_y_inf = df_inflams["yy"].values
        preds_inf = df_inflams["inflams"].values

        coords_y_max = max(coords_y_tum.max()+2*step,coords_y_inf.max()+2*step_inflam)
        coords_x_max = max(coords_x_tum.max()+2*step,coords_x_inf.max()+2*step_inflam)
        # init black image
        image_bin_tum = np.zeros(
            (int(coords_y_max), int(coords_x_max)), dtype=np.uint8
        )

        # make white pixels where tumor
        set_p = set()
        for x, y, p in zip(coords_x_tum, coords_y_tum, preds):
            set_p.add(p)
            if p in [1, 2]:
                image_bin_tum[int(y) : int(y) + step, int(x) : int(x) + step] = 1

        # init black image for inflamation
        image_inf = np.zeros(
            (int(coords_y_max), int(coords_x_max)), dtype=np.uint8
        )

        # make pixels where value
        set_inflams = set()
        for x, y, p in zip(coords_x_inf, coords_y_inf, preds_inf):
            set_inflams.add(p)
            if p >0:
                image_inf[int(y) : int(y) + step_inflam, int(x) : int(x) + step_inflam] =p

        if verbose:
            plt.subplot(2,2,1)
            plt.imshow(image_inf,vmax=5)
            plt.title('image inflammation')
            plt.subplot(2,2,2)
            plt.imshow(image_bin_tum)
            plt.title('image tumor')
            plt.subplot(2,2,3)
            img_tum_res = image_bin_tum
            plt.imshow(img_tum_res)
            plt.title('resized tumor')
            plt.subplot(2,2,4)
            plt.imshow((img_tum_res*image_inf),vmax=5)
            plt.title("sum")
            plt.savefig(inflams_verbose+'/'+slide.replace('pt','png'))
        
        image_tum_res = cv2.copyMakeBorder(
            img_tum_res,
            top=padding,
            bottom=padding,
            left=padding,
            right=padding,
            borderType=cv2.BORDER_CONSTANT,
            value=[0],
        )

        # fill in the holes 
        # TODO: adapt kernel depending on the mpp
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (19, 19)
        )  # rectangular kernel. 
        closed_image = cv2.morphologyEx(image_tum_res, cv2.MORPH_CLOSE, kernel)

        # remove small objects
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (35, 35)
        )  # rectangular kernel corresponds to 1 cm
        opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, kernel)

        # take external border 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (55, 55)) # corresponds to 1.6 cm
        dilated_image = cv2.dilate(opened_image, kernel, iterations=1)
        out_tumor = dilated_image - opened_image

        # take internal border
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (55, 55)) # corresponds to 1.6 cm
        eroded_image = cv2.erode(opened_image, kernel, iterations=1)
        in_tumor = opened_image - eroded_image

        # Define colors for each version: image_bin (red), eroded_image (blue), dilated_image (red)
        image_bin_color = np.stack(
            [opened_image, np.zeros_like(opened_image), np.zeros_like(opened_image)], axis=2
        )  # red channel (255,0,0)

        eroded_image_color = np.stack(
            [np.zeros_like(in_tumor), in_tumor, np.zeros_like(in_tumor)], axis=2
        )  # green channel (0, 255, 0)

        dilated_image_color = np.stack(
            [np.zeros_like(out_tumor), np.zeros_like(out_tumor), out_tumor], axis=2
        )  # blue channel (0, 0, 255)

        colored_in_tumor = np.clip(image_bin_color + eroded_image_color, 0, 255).astype(
            np.uint8
        )
        
        colored_out_tumor = np.clip(image_bin_color + dilated_image_color, 0, 255).astype(
            np.uint8
        )
        
        eroded_image_color = np.stack(
            [in_tumor, in_tumor, np.zeros_like(in_tumor)], axis=2
        )  # green channel (0, 255, 0)
        
        colored_in_out = np.clip(eroded_image_color + dilated_image_color, 0, 255).astype(
            np.uint8
        )


        if verbose:
            plt.subplot(2,2,1)
            plt.imshow(img_tum_res )
            plt.title('colored in tumor')
            plt.subplot(2,2,2)
            plt.imshow(closed_image )
            plt.title('colored in tumor')
            plt.subplot(2,2,3)
            plt.imshow(img_tum_res)
            plt.title('resized tumor')
            plt.subplot(2,2,4)
            plt.imshow(colored_in_out
                       )
            plt.title("sum")
            plt.savefig(inflams_verbose+'/'+slide.replace('.pt','_inout.png'))

        black_pixels_mask = np.all(image_bin_color == [0, 0, 0], axis=-1)
        image_bin_color[black_pixels_mask] = [255, 255, 255]

        in_tumor_mask = cv2.cvtColor(eroded_image_color, cv2.COLOR_RGB2GRAY)
        out_tumor_mask = cv2.cvtColor(dilated_image_color, cv2.COLOR_RGB2GRAY)
        inout_tumor_mask = in_tumor_mask + out_tumor_mask

        final_inflam_tumor_image = cv2.bitwise_and(
            image_inflams,
            image_inflams,
            mask=opened_image,
        )
        final_inout_tumor_image = cv2.bitwise_and(
            image_inflams,
            image_inflams,
            mask=inout_tumor_mask,
        )

        # compute number of inflams in tumor
        idx_X, idx_Y = np.nonzero(final_inflam_tumor_image)
        INFLAM_IN_ALL_T = final_inflam_tumor_image.sum() / len(idx_X)
        # print("inflam cells inside all tumor (mean per patch)", INFLAM_IN_ALL_T)

        # compute number of inflams inout tumor
        idx_X, idx_Y = np.nonzero(final_inout_tumor_image)
        INFLAM_INOUT_T = final_inout_tumor_image.sum() / len(idx_X)
        # print("inflam cells surrounding tumor (in & out) (mean per patch)", INFLAM_INOUT_T)

        # add to the table
        df.loc[df["lame"] == slide_name] = [slide_name, INFLAM_INOUT_T, INFLAM_IN_ALL_T]

        df["patient"] = df["lame"].apply(lambda x: x[:-1]).astype(int)

    df_inflams = pd.DataFrame(
        index=df["patient"].unique(),
        columns=["patient", "peri-tumoral", "intra-tumoral"],
    )
    
    # compute mean
    df_inflams["patient"] = df["patient"].unique()
    for patient in df["patient"].unique():
        df_inflams.loc[df_inflams["patient"] == patient] = [patient] + list(
            df.loc[df["patient"] == patient][["peri-tumoral", "intra-tumoral"]].mean()
        )


    df_inflams.to_csv("data/tabs/final_inflams_features.csv", index=False)

if __name__ == "__main__":
    main()
