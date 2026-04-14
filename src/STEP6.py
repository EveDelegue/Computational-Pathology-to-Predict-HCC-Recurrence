import torch
import cv2
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from utils.utils_tumor import (
    gen_image_from_coords,
    gen_image_from_coords_bis,
)



def main():
    # read hyperparameters
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    patch_size = config["patching"]["patch_size_dict"]["PB"]
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
    step = int(vis_scale * patch_size)
    padding = 2 * step

    tumor_checkpoints =  config["paths"]["pth_to_tumor_ckpts"]
    coords_checkpoints =  config["paths"]["pth_to_coords"]
    inflams_checkpoints =  config["paths"]["pth_to_inflams_ckpts"]


    # create a dataframe
    cols = ["lame", "peri-tumoral", "intra-tumoral"]
    df = pd.DataFrame(columns=cols)
    # fill the "lame" column
    df["lame"] = [
        slide_name.split("_")[0] for slide_name in os.listdir(inflams_checkpoints)
    ]
    slides = sorted(os.listdir(inflams_checkpoints))

    # for each lame,
    for slide in slides:
        slide_name = slide.split("_")[0]
        hospital = slide.split("_")[1]
        # load the tumoral predictions
        chkpt_coords = torch.load(f"{coords_checkpoints}/{slide_name}_{hospital}_coords_checkpoint.pt")
        chkpt_tumor = torch.load(
            f"{tumor_checkpoints}/{slide_name}_{hospital}_preds_probas_checkpoint.pt"
        )
        chkpt_inflam = torch.load(
            f"{inflams_checkpoints}/{slide_name}_{hospital}_coords_inflams_checkpoint.pt"
        )

        # new pd frame for tumor
        df_tumor = pd.DataFrame()
        # copy from the loaded ones
        df_tumor["x"] = chkpt_tumor["coords_x"]
        df_tumor["y"] = chkpt_tumor["coords_y"]
        df_tumor["tumor"] = [p.item() for p in chkpt_tumor["har_mean_preds"]]

        # new pd frame for inflam
        df_inflams = pd.DataFrame()
        df_inflams["x"] = chkpt_inflam["coords_x"]
        df_inflams["y"] = chkpt_inflam["coords_y"]
        df_inflams["inflams"] = chkpt_inflam["inflams"]

        [x_start, y_start, _, _] = chkpt_coords["xy_start_end"]
        
        ###### ATTENTION ! PROBLEMES ! ######
        df_map = pd.DataFrame(
            columns=["x", "y", "tumor", "inflams"], index=range(len(df_tumor))
        )
        for j in range(len(df_tumor)):
            x, y, p = df_tumor.iloc[j].values
            _, _, ii = df_inflams.loc[
                (df_inflams["x"] == x) & (df_inflams["y"] == y)
            ].values[0]
            df_map.iloc[j] = pd.Series([x, y, p, ii], index=df_map.columns)

        ###### fin des problèmes ######

        df_map["xx"] = df_map["x"] * vis_scale - x_start
        df_map["yy"] = df_map["y"] * vis_scale - y_start

        coords_x = df_map["xx"].values
        coords_y = df_map["yy"].values
        preds = df_map["tumor"].values

        # init black image
        image_bin = np.zeros(
            (int(coords_y.max()) + 2 * step, int(coords_x.max()) + 2 * step), dtype=np.uint8
        )

        # make white pixels where tumor
        set_p = set()
        for x, y, p in zip(coords_x, coords_y, preds):
            set_p.add(p)
            if p in [1, 2]:
                image_bin[int(y) : int(y) + step, int(x) : int(x) + step] = 255

        image_bin = cv2.copyMakeBorder(
            image_bin,
            top=padding,
            bottom=padding,
            left=padding,
            right=padding,
            borderType=cv2.BORDER_CONSTANT,
            value=[0],
        )

        image_inflams = gen_image_from_coords_bis(
            df_map["xx"],
            df_map["yy"],
            df_map["inflams"].values,
            step,
        )

        scaled_slide = cv2.copyMakeBorder(
            chkpt_tumor["scaled_slide"].astype(np.uint8),
            top=padding,
            bottom=padding,
            left=padding,
            right=padding,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )

        image_tumor = gen_image_from_coords(
            df_map["xx"],
            df_map["yy"],
            [torch.tensor(e) for e in df_map["tumor"]],
            step,
            colors,
        )
        image_tumor = cv2.copyMakeBorder(
            image_tumor,
            top=padding,
            bottom=padding,
            left=padding,
            right=padding,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )

        image_inflams = cv2.copyMakeBorder(
            image_inflams,
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
        )  # 5x5 rectangular kernel
        closed_image = cv2.morphologyEx(image_bin, cv2.MORPH_CLOSE, kernel)

        # remove small objects
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (35, 35)
        )  # 5x5 rectangular kernel
        opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, kernel)

        # take external border 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (55, 55))
        dilated_image = cv2.dilate(opened_image, kernel, iterations=1)
        out_tumor = dilated_image - opened_image

        # take internal border
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (55, 55))
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
