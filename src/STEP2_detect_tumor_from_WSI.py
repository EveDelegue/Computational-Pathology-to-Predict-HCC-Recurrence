import torch
import os
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from utils.ImageSet import ImageSet_2
from utils.utils_tumor import (
    load_models,
    get_pred_proba_multi_2,
    gen_image_from_coords,
    gen_multiscale_patches,
    compute_mean_predictions,
)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide_name", type=str)
    parser.add_argument("--verbose",type=bool,default=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    slide_name = args.slide_name.split("/")[-1] # ex : data/patches/93A_PB -> 93A_PB
    verbose = args.verbose

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    tumor_checkpoints = config["paths"]["pth_to_tumor_ckpts"]
    if not os.path.exists(
        f"{tumor_checkpoints}/{slide_name}_preds_probas_checkpoint.pt" # si l'étape n'a pas été faite
    ):  
        # load params
        patches_dir = config["paths"]["pth_to_patches"]
        coords_checkpoints = config["paths"]["pth_to_coords"]
        preds_wsis_results = config["paths"]["pth_to_preds_wsis"]

        hospital = slide_name.split('_')[-1] # ex : 93A_PB -> PB
        patch_size_dict = config["patching"]["patch_size_dict"]
        if hospital not in patch_size_dict.keys():
            raise KeyError(f"no resolution defined for this hospital : {hospital}. Check that the hospital's name in the config file is the same as in the data folder.")

        patch_size = patch_size_dict[hospital]
        batch_size = config["model"]["batch_size"]

        vis_scale = config["patching"]["vis_scale"]
        
        colors = {
            0: config["visualization"]["colors"]["healthy"],
            1: config["visualization"]["colors"]["non_pej"],
            2: config["visualization"]["colors"]["pej"],
        }
        colors_TNT = {
            0: (34, 139, 34),
            1: (178, 34, 34),
            2: (178, 34, 34),
        }

        cmap = plt.get_cmap(config["visualization"]["color_map"])
        # load tumor prediction models
        models = load_models(pth=config["model"]["path_triple_resnets"]) # were they trained on an external dataset ?

        # color norm and multiscale patches
        if hospital in ["PB", "HM"]:
            print("patches are from Paul-Brousse ==> no color transfer needed")
            X, _ = gen_multiscale_patches(slide_name, patches_dir)
        else:
            reference_pb = plt.imread("notebooks/HES__5.jpeg")
            print("patches are not from Paul-Brousse ==> color transfer is needed")
            X, _ = gen_multiscale_patches(
                slide_name,
                patches_dir,
                apply_clr_transfer=True,
                reference=reference_pb,
            )

        # inference on all tiles
        Data = ImageSet_2(X, tt.Compose([tt.ToTensor()]))
        loader = DataLoader(Data, batch_size=batch_size, shuffle=False)
        all_y_probas = []
        with torch.no_grad():
            for i, m in enumerate(models):
                print(f"model {i+1}", end=" ")
                _, _, y_proba = get_pred_proba_multi_2(m, loader)
                all_y_probas.append(y_proba)

        # ensembling
        y_arith_mean_proba, y_arith_preds = compute_mean_predictions(
            all_y_probas, "arithmetic"
        )
        y_geo_mean_proba, y_geo_preds = compute_mean_predictions(
            all_y_probas, "geometric"
        )
        y_har_mean_proba, y_har_preds = compute_mean_predictions(
            all_y_probas, "harmonic"
        )

        # save prediction
        coords_x, coords_y = [], []
        for patch in os.listdir(f"{patches_dir}/{slide_name}"):
            _, _, x, _, y = patch[:-4].split("_")
            coords_x.append(int(x))
            coords_y.append(int(y))

        
        coords = torch.load(
            f"{coords_checkpoints}/{slide_name}_coords_checkpoint.pt",
            weights_only=False,
        )
        scaled_slide = coords["scaled_slide"]
        [x_start, y_start, _, _] = coords["xy_start_end"]

        to_save = {
            "scaled_slide": scaled_slide,
            "coords_x": coords_x,
            "coords_y": coords_y,
            "har_mean_proba": y_har_mean_proba,
            "har_mean_preds": y_har_preds,
            "geo_mean_proba": y_geo_mean_proba,
            "geo_mean_preds": y_geo_preds,
            "arith_mean_proba": y_arith_mean_proba,
            "arith_mean_preds": y_arith_preds,
        }
        handle = f"{tumor_checkpoints}/{slide_name}_preds_probas_checkpoint.pt" # ex : "checkpoints/tumor_checkpoints/93A_PB_preds_probas_checkpoint.pt
        torch.save(to_save, handle)

        # plot prediction
        if verbose:
            coords_x = np.array(coords_x) * vis_scale - x_start
            coords_y = np.array(coords_y) * vis_scale - y_start

            pred_sets = [
                ("ArithMean", y_arith_preds, y_arith_mean_proba),
                ("GeoMean", y_geo_preds, y_geo_mean_proba),
                ("HarMean", y_har_preds, y_har_mean_proba),
            ]
            fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(24, 14))
            fig.suptitle(f"{slide_name}_preds_NT_NP_P", fontsize=18)
            for j, (title, preds, probas) in enumerate(pred_sets, start=1):
                class_colors = [np.array(colors[int(p.item())]) / 255.0 for p in preds]
                prob_values = [p.max().item() for p in probas]
                ax[0, j].imshow(scaled_slide)
                ax[0, j].scatter(coords_x, coords_y, c=class_colors, marker="s")
                ax[0, j].set_title(f"Preds with {title}")
                ax[0, j].axis("off")
                ax[1, j].imshow(scaled_slide)
                sc = ax[1, j].scatter(
                    coords_x, coords_y, c=prob_values, marker="s", cmap=cmap
                )
                ax[1, j].axis("off")
                cbar = fig.colorbar(sc, ax=ax[1, j], fraction=0.046, pad=0.04)
                cbar.set_label("Confidence level (Proba)", rotation=270, labelpad=15)
            ax[0, 0].imshow(scaled_slide)
            ax[0, 0].axis("off")
            ax[1, 0].axis("off")
            plt.tight_layout()
            plt.savefig(f"{preds_wsis_results}/{slide_name}_preds_NT_NP_P.png")

            step = int(vis_scale * patch_size)

            image_arith = gen_image_from_coords(
                coords_x, coords_y, y_arith_preds, step, colors
            )
            median_filtered = Image.fromarray(image_arith).filter(
                ImageFilter.MedianFilter(size=2 * step + 1)
            )
            mode_filtered = Image.fromarray(image_arith).filter(
                ImageFilter.ModeFilter(size=2 * step + 1)
            )

            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
            fig.suptitle(f"{slide_name}_preds_NT_NP_P_smoothed_Arith", fontsize=18)
            axes[0].imshow(scaled_slide)
            axes[0].set_title("preds")
            axes[0].axis("off")
            axes[1].imshow(median_filtered)
            axes[1].set_title("median")
            axes[1].axis("off")
            axes[2].imshow(mode_filtered)
            axes[2].set_title("mode")
            axes[2].axis("off")
            plt.tight_layout()
            plt.savefig(
                f"{preds_wsis_results}/{slide_name}_preds_NT_NP_P_smoothed_Arith.png"
            )

            image_geo = gen_image_from_coords(
                coords_x, coords_y, y_geo_preds, step, colors_TNT
            )
            median_filtered = Image.fromarray(image_geo).filter(
                ImageFilter.MedianFilter(size=2 * step + 1)
            )  #
            mode_filtered = Image.fromarray(image_geo).filter(
                ImageFilter.ModeFilter(size=2 * step + 1)
            )

            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
            fig.suptitle(f"{slide_name}_preds_NT_T_smoothed_Geo", fontsize=18)
            axes[0].imshow(image_geo)
            axes[0].set_title("preds")
            axes[0].axis("off")
            axes[1].imshow(median_filtered)
            axes[1].set_title("median")
            axes[1].axis("off")
            axes[2].imshow(mode_filtered)
            axes[2].set_title("mode")
            axes[2].axis("off")
            plt.tight_layout()
            plt.savefig(f"{preds_wsis_results}/{slide_name}_preds_NT_T_smoothed_Geo.png")
    else:
        print(slide_name, "already processed")


if __name__ == "__main__":
    main()
