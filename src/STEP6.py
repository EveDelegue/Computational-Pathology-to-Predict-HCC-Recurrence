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
from utils.utils import draw_contours

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
    mpp_dict = config["patching"]["mpp_dict"]

    patch_size = config["patching"]["patch_size_dict"]["PB"]
    tia_patch_size = config["patching"]["tia_patch_size"]

    vis_scale = config["patching"]["vis_scale"]

    step = int(vis_scale * patch_size)

    step_inflam = int(vis_scale * tia_patch_size)

    tumor_checkpoints =  config["paths"]["pth_to_tumor_ckpts"]
    coords_checkpoints =  config["paths"]["pth_to_coords"]
    inflams_checkpoints =  config["paths"]["pth_to_inflams_ckpts"]
    inflams_verbose = config["paths"]["pth_to_inflams_wsis"]


    # create a dataframe
    cols = ["lame", "peri-tumoral", "intra-tumoral",'hospital']
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
        aspect_ratio = mpp_dict[hospital]/mpp_dict["PB"]
        # load the tumoral predictions
        chkpt_coords = torch.load(f"{coords_checkpoints}/{slide_name}_{hospital}_coords_checkpoint.pt",weights_only=False)
        chkpt_tumor = torch.load(
            f"{tumor_checkpoints}/{slide_name}_{hospital}_preds_probas_checkpoint.pt",weights_only=False
        )
        chkpt_inflam = torch.load(
            f"{inflams_checkpoints}/{slide_name}_{hospital}_coords_inflams_checkpoint.pt",weights_only=False
        )
        scaled_slide = chkpt_tumor["scaled_slide"]
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
        image_bin = np.zeros(
            (int(coords_y_max), int(coords_x_max)), dtype=np.uint8
        )
        # make white pixels where tumor
        set_p = set()
        for x, y, p in zip(coords_x_tum, coords_y_tum, preds):
            set_p.add(p)
            # create a binary mask for tissue
            image_bin[int(y) : int(y) + step, int(x) : int(x) + step] = 1
            if p in [1, 2]:
                # create a binary mask for tumor
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
            plt.figure()
            plt.subplot(1,3,1)
            plt.imshow(image_inf,vmax=5)
            plt.title('image inflammation')
            plt.subplot(1,3,2)
            plt.imshow(image_bin_tum)
            plt.title('image tumor')
            img_tum_res = image_bin_tum
            plt.subplot(1,3,3)
            plt.imshow((img_tum_res*image_inf),vmax=5)
            plt.title("sum")
            plt.savefig(inflams_verbose+'/'+slide.replace('_coords_inflams_checkpoint.pt','tum_inf.png'))
        
        # fill in the holes 
        size = 2 * int((19 /aspect_ratio)/2) + 1 # closest odd number to this 
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (size, size)
        )  # rectangular kernel. 
        # clean tissue mask
        closed_image_og = cv2.morphologyEx(image_bin, cv2.MORPH_CLOSE, kernel)
        # clean tumor mask
        closed_image = cv2.morphologyEx(img_tum_res, cv2.MORPH_CLOSE, kernel)

        # remove small objects
        size = 2 * int((35 /aspect_ratio)/2) + 1 # closest odd number to this 
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (size, size)
        )  # rectangular kernel corresponds to 1 cm
        # clean tissue mask
        opened_image_og = cv2.morphologyEx(closed_image_og, cv2.MORPH_OPEN, kernel)
        # clean tumor mask
        opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, kernel)

        # take external border
        size = 2 * int((27 /aspect_ratio)/2) + 1 # closest odd number to this  
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size)) # corresponds to 1 mm 
        # tissue + contour
        dilated_image_og = cv2.dilate(opened_image_og, kernel, iterations=1)
        # tumor + contour
        dilated_image = cv2.dilate(opened_image, kernel, iterations=1)
        # tumor contour
        out_tumor = dilated_image - opened_image
        # tissue contour
        out_og = dilated_image_og-closed_image_og
        # tumor contour that is NOT in the tissue contour
        out_tumor = out_tumor.astype(bool) * (1-out_og).astype(bool)
        out_tumor = out_tumor.astype(np.uint8)

        # take internal border
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size)) # corresponds to 1 mm
        # inside of the tissue
        eroded_image_og = cv2.erode(closed_image_og, kernel, iterations=1)
        # inside of the tumor
        eroded_image = cv2.erode(opened_image, kernel, iterations=1)
        # tumor contour
        in_tumor = opened_image - eroded_image
        # tissue contour
        in_og = closed_image_og - eroded_image_og
        # tumor contour that is NOT in the tissue internal contour
        in_tumor = in_tumor.astype(bool) * (1-in_og).astype(bool)
        in_tumor = in_tumor.astype(np.uint8)

        inout_tumor = in_tumor.astype(bool) | out_tumor.astype(bool)
        inout_tumor = inout_tumor.astype(np.uint8)
        final_inflam_tumor_image = image_inf*eroded_image
        final_inout_tumor_image = image_inf*(inout_tumor)

        if verbose:
            plt.figure()
            plt.subplot(2,3,1)
            plt.imshow(draw_contours(img_tum_res, scaled_slide) )
            plt.title('original image')
            plt.subplot(2,3,2)
            plt.imshow(draw_contours(opened_image, scaled_slide))
            plt.title("Clean contours")
            plt.subplot(2,3,3)
            plt.imshow(draw_contours(opened_image_og, scaled_slide))
            plt.title("tissue contour")
            plt.subplot(2,3,4)
            plt.imshow(draw_contours(eroded_image, scaled_slide))
            plt.title("in tumor")
            plt.subplot(2,3,5)
            plt.imshow(draw_contours(inout_tumor, scaled_slide))
            plt.title("tumor peripheral")
            plt.subplot(2,3,6)
            plt.imshow(final_inout_tumor_image,vmax=2)
            plt.title("final inflammation contour")
            plt.tight_layout()
            plt.savefig(inflams_verbose+'/'+slide.replace('_coords_inflams_checkpoint.pt','_inout.png'))


        

        # compute number of inflams in tumor
        idx_X, idx_Y = np.nonzero(final_inflam_tumor_image)
        INFLAM_IN_ALL_T = final_inflam_tumor_image.sum() / len(idx_X)
        # print("inflam cells inside all tumor (mean per patch)", INFLAM_IN_ALL_T)

        # compute number of inflams inout tumor
        idx_X, idx_Y = np.nonzero(final_inout_tumor_image)
        INFLAM_INOUT_T = final_inout_tumor_image.sum() / len(idx_X)
        # print("inflam cells surrounding tumor (in & out) (mean per patch)", INFLAM_INOUT_T)

        # add to the table
        df.loc[df["lame"] == slide_name] = [slide_name, INFLAM_INOUT_T, INFLAM_IN_ALL_T,hospital]

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
