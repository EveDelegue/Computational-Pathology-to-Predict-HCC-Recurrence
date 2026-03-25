import torch
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from utils.utils_tumor import (
    gen_image_from_coords,
    get_RdYlGr_masks,
    get_largest_connected_area,
    pej_color,
    non_pej_color,
    healthy_color,
)

def main():

    color2class = {
        tuple(healthy_color): "healthy",
        tuple(non_pej_color): "non pej",
        tuple(pej_color): "pej",
    }

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    patch_size = config["patching"]["patch_size_dict"]["PB"]
    colors = {
        0: config["visualization"]["colors"]["healthy"],
        1: config["visualization"]["colors"]["non_pej"],
        2: config["visualization"]["colors"]["pej"],
    }

    vis_scale = config["patching"]["vis_scale"]
    step = int(vis_scale * patch_size)
    tumor_checkpoints =  config["paths"]["pth_to_tumor_ckpts"]


    colors_NT_NP_P = {
        0: np.array(healthy_color) / 255,
        1: np.array(non_pej_color) / 255,
        2: np.array(pej_color) / 255,
    }

    cols = [
        "lame",
        "#NT",
        "#NP",
        "#P",
        "#pixels",
        "NT_CntArea",
        "NP_CntArea",
        "P_CntArea",
        "slide_w",
        "slide_h",
        "patch_size",
    ]

    df = pd.DataFrame(columns=cols)
    df["lame"] = [slide_name.split("_")[0] for slide_name in os.listdir(tumor_checkpoints)]
    slides = sorted(os.listdir(tumor_checkpoints))
    for slide_name_0 in slides:
        slide_name = slide_name_0.split("_")[0]
        hospital = slide_name_0.split("_")[1]
        patch_size = 1152
        if len(slide_name) > 4:
            n = int(slide_name[:3])
            if 110 < n <= 160:
                patch_size = 626
            elif 160 < n < 213 or 223 <= n < 253:
                patch_size = 1094
        with open(
            f"{tumor_checkpoints}/{slide_name}_{hospital}_preds_probas_checkpoint.pt", "rb"
        ) as handle:
            tumor_data = torch.load(handle)
        y_har = tumor_data["arith_mean_preds"]

        with open(
            f"checkpoints/coords_checkpoints/{slide_name}_{hospital}_coords_checkpoint.pt",
            "rb",
        ) as handle:
            coords = torch.load(handle)

        coords_x, coords_y = [], []
        for patch in os.listdir(f"data/patches/{slide_name}_{hospital}"):
            _, _, x, _, y = patch[:-4].split("_")
            coords_x.append(int(x))
            coords_y.append(int(y))
        coords["coords_x"], coords["coords_y"] = coords_x, coords_y
        [x_start, y_start, _, _] = coords["xy_start_end"]
        [_, _, real_w, real_h] = coords["xywh_real"]
        coords_x = np.array(coords_x) * vis_scale - x_start
        coords_y = np.array(coords_y) * vis_scale - y_start

        image = gen_image_from_coords(coords_x, coords_y, y_har, step, colors)
        masked_image_rd, masked_image_yl, masked_image_gr = get_RdYlGr_masks(image)
        (_, non_pej_area) = get_largest_connected_area(
            masked_image_yl, non_pej_color
        )
        (_, pej_area) = get_largest_connected_area(masked_image_rd, pej_color)
        (_, healthy_area) = get_largest_connected_area(
            masked_image_gr, healthy_color
        )
        N = image.shape[0] * image.shape[1]
        NonWhite_image = [
            tuple(pixel)
            for pixel in image.reshape(N, 3)
            if (pixel != (255, 255, 255)).any()
        ]
        count = [
            (dict(Counter(NonWhite_image))[e] if e in dict(Counter(NonWhite_image)) else 0)
            for e in color2class
        ]
        df.loc[df["lame"] == slide_name] = (
            [slide_name]
            + count
            + [len(NonWhite_image)]
            + [healthy_area, non_pej_area, pej_area]
            + [real_w, real_h, patch_size]
        )

        df["%P"] = df["#P"] / (df["#NP"] + df["#P"])
    df["NP_CntArea_norm"] = (df["NP_CntArea"] * df["patch_size"] ** 2) / (
        df["slide_w"] * df["slide_h"]
    )
    df["P_CntArea_norm"] = (df["P_CntArea"] * df["patch_size"] ** 2) / (
        df["slide_w"] * df["slide_h"]
    )
    df["patient"] = df["lame"].apply(lambda x: x[:-1]).astype(int)
    final_features = ["patient", "lame", "%P", "P_CntArea_norm", "NP_CntArea_norm"]
    df = df[final_features]
    df.head(10)

    df_max = pd.DataFrame(
        index=df["patient"].unique(),
        columns=[
            "patient",
            "%P_max",
            "P_CntArea_norm_max",
        ],
    )
    df_max["patient"] = df["patient"].unique()
    for patient in df["patient"].unique():
        df_max.loc[df_max["patient"] == patient] = [patient] + list(
            df.loc[df["patient"] == patient][["%P", "P_CntArea_norm"]].max()
        )
    df_max[["%P_max", "P_CntArea_norm_max"]] = df_max[
        ["%P_max", "P_CntArea_norm_max"]
    ].astype("float")

    df_mean = pd.DataFrame(
        index=df["patient"].unique(),
        columns=["patient", "%P", "NP_CntArea_norm", "P_CntArea_norm"],
    )
    df_mean["patient"] = df["patient"].unique()
    for patient in df["patient"].unique():
        df_mean.loc[df_mean["patient"] == patient] = [patient] + list(
            df.loc[df["patient"] == patient][
                ["%P", "NP_CntArea_norm", "P_CntArea_norm"]
            ].mean()
        )
    df_mean[["%P", "NP_CntArea_norm", "P_CntArea_norm"]] = df_mean[
        ["%P", "NP_CntArea_norm", "P_CntArea_norm"]
    ].astype("float")


    df_ai_final = pd.merge(
        df_mean,
        df_max,
        on="patient",
        how="inner",
    )

    df_ai_final.head()

    df_ai_final.to_csv("data/tabs/final_tumor_features.csv", index=False)

if __name__ == "__main__":
    main()
