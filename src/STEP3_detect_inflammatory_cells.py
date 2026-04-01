import os
import joblib
import torch
import argparse
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from utils.utils_inflams import (
    color_dict_pannuke,
    get_Inflammatory,
)
import multiprocessing

device = "cuda" if torch.cuda.is_available() else "cpu"
print("running on", device)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide_name", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--verbose",type=bool,default=True)
    parser.add_argument(
        "--hovernet", type=str, default="pannuke"
    )  # "pannuke", "monusac", "consep"
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    verbose = args.verbose
    slide_name = args.slide_name.split(os.path.sep)[-1] #ex : 93A_PB
    batch_size = args.batch_size
    hovernet = args.hovernet

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    vis_scale = config["patching"]["vis_scale"]
    pth_to_inflams_dats = config["paths"]["pth_to_inflams_dats"]
    patches_dir = config["paths"]["pth_to_patches_bis"]
    coords_checkpoints = config["paths"]["pth_to_coords"]
    inflams_checkpoints = config["paths"]["pth_to_inflams_ckpts"]
    inflams_wsis_results = config["paths"]["pth_to_inflams_wsis"]

    if not os.path.exists(
        f"{inflams_checkpoints}/{slide_name}_coords_inflams_checkpoint.pt"
    ):
        # load coords
        coords = torch.load(
            f"{coords_checkpoints}/{slide_name}_coords_checkpoint.pt",
            weights_only=False,
        )

        coords_x, coords_y = [], []
        for patch in os.listdir(f"{patches_dir}/{slide_name}"):
            _, _, x, _, y = patch[:-4].split("_")
            coords_x.append(int(x))
            coords_y.append(int(y))

        scaled_slide = coords["scaled_slide"]
        [x_start, y_start, _, _] = coords["xy_start_end"]
        coords_x = np.array(coords_x) * vis_scale - x_start
        coords_y = np.array(coords_y) * vis_scale - y_start

        # init model
        if hovernet in ["pannuke", "monusac"]:
            model = NucleusInstanceSegmentor(
                model="hovernet_fast-" + hovernet, batch_size=batch_size, 
                device=device, num_workers=multiprocessing.cpu_count()//2
            )
        else:
            model = NucleusInstanceSegmentor(
                model="hovernet_original-consep", batch_size=batch_size,
                device=device, num_workers=multiprocessing.cpu_count()//2
            )

        save_dir = f"{pth_to_inflams_dats}/{slide_name}/" # ex : checkpoints/inflam_dats/93A_PB
        images = [
            f"{patches_dir}/{slide_name}/{e}"
            for e in os.listdir(f"{patches_dir}/{slide_name}")
        ]

        # detect nucleus
        output = model.run(
            images,
            save_dir=save_dir,
            input_resolutions=[{"units": "baseline", "resolution": 1.0}],
            auto_get_mask=False,
        )



        # load the output data
        file_map = joblib.load(f"{save_dir}/file_map.dat") # ex : checkpoints/inflam_dats/93A_PB/file_map.dat
        dict4 = {}  # dictionary of results: {id: centroid, contour, proba, type}
        dict_xy_nucs = {}  # dictionary {x,y): [nuc_id1, nuc_id2,...]}
        

        for im_dir, out_dir in tqdm(
            file_map, desc=f"Processing {save_dir}/file_map.dat", leave=False
        ):
            out_dir = f"{save_dir}/{out_dir.split('/')[-1]}.dat" # ex : checkpoints/inflam_dats/93A_PB/0.dat
            tile_preds = joblib.load(f"{out_dir}")
            InfLymph_nucleus1, _ = get_Inflammatory(tile_preds, color_dict_pannuke) # get a detail of all inflamatory nucleus
            dict4 |= InfLymph_nucleus1
            _, x, _, y = im_dir[:-4].split("_")[-4:] 
            dict_xy_nucs[(x, y)] = list(tile_preds.keys()) # dict of all the inflammatory nucleus ids
        final_dict = defaultdict(list)
        for xy, v in dict_xy_nucs.items():
            final_dict[xy].extend([nuc_id for nuc_id in v if nuc_id in dict4])

        coords_x, coords_y, num_nucleus = zip(
            *((int(k[0]), int(k[1]), len(v)) for k, v in final_dict.items())
        )
        inf_nucleus_sorted, coords_X, coords_Y = zip(
            *sorted(zip(num_nucleus, coords_x, coords_y), key=lambda x: x[0])
        )

        to_save = {
            "scaled_slide": scaled_slide,
            "coords_x": coords_X,
            "coords_y": coords_Y,
            "inflams": inf_nucleus_sorted,
            "vis_scale": vis_scale,
            "xy_start_end": coords["xy_start_end"],
            "xywh_real": coords["xywh_real"],
        }
        handle = f"{inflams_checkpoints}/{slide_name}_coords_inflams_checkpoint.pt" # ex : checkpoints/inflam_checkpoints/93A_PB_coords_inflams_checkpoint.pt
        torch.save(to_save, handle)

        # plot inflam map
        if verbose : 
            cX = np.array(coords_X) * vis_scale - x_start
            cY = np.array(coords_Y) * vis_scale - y_start

            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
            fig.suptitle(f"{slide_name} based on Inflammatory nucleus", fontsize=20)
            axes[0].imshow(scaled_slide)
            axes[0].axis("off")
            axes[1].imshow(scaled_slide)
            sc = axes[1].scatter(
                cX, cY, c=np.log1p(inf_nucleus_sorted), cmap="turbo", marker="s"
            )
            axes[1].axis("off")
            cbar = fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04)
            cbar.set_label(
                "number of inflammatory nucleus (LOG SCALE)", rotation=270, labelpad=15
            )
            plt.tight_layout()
            plt.savefig(f"{inflams_wsis_results}/{slide_name}_inflammation_map.png")

    else:
        print(slide_name, "already processed (inflam detection)")


if __name__ == "__main__":
    main()
