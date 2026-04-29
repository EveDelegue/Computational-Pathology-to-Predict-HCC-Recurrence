import torch
import argparse
import yaml
import os
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils.utils_inflams import load_net, PatchDataset, inference, post_process

device = "cuda" if torch.cuda.is_available() else "cpu"
print("running on", device)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide_name", type=str,default='data/patches_bis/93A_PB')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--verbose",type=bool,default=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    verbose = args.verbose
    slide_name = args.slide_name.split(os.path.sep)[-1] #ex : 93A_PB
    batch_size = args.batch_size

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    vis_scale = config["patching"]["vis_scale"]
    patches_dir = config["paths"]["pth_to_patches_bis"]
    coords_checkpoints = config["paths"]["pth_to_coords_bis"]
    pth_to_inflams_dats = config["paths"]["pth_to_inflams_dats"]
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

        ###### inflam detection
        # load the net
        net = load_net(device=device)

        # make dataloader
        dataset = PatchDataset(os.path.join(patches_dir,slide_name))
        dataloader = DataLoader(dataset=dataset,batch_size=batch_size)

        with torch.no_grad():  # dont compute gradient
            # inference
            raw_results = inference(dataloader,net)
            torch.save(raw_results,os.path.join(pth_to_inflams_dats,f'{slide_name}_raw.pt'))
        
        # post processing
        
        num_nucleus,coords_x,coords_y = post_process(raw_results)

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




