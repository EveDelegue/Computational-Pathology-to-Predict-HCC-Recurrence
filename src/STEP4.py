import torch
import os
import yaml
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import pandas as pd

from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from utils.utils_nucleus import get_contours_2, vectorize,pinv, getHstain, segmentNucleus, computeFeatures
from utils.PGA import PGA
import joblib



device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide_name", type=str)
    parser.add_argument("--verbose",type=bool,default=True)
    args = parser.parse_args()
    return args

def main():

    # load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    hemato_1 = config["staining"]["hemato_1"]
    eosin = config["staining"]["eosin"]
    safran = config["staining"]["safran"]
    hemato_2 = config["staining"]["hemato_2"]

    vis_scale = config["patching"]["vis_scale"]
    tumor_checkpoints = config["paths"]["pth_to_tumor_ckpts"]
    patches_path = config["paths"]["pth_to_patches_bis"]
    save_dir = config["paths"]["pth_to_nuc_dats"]

    path_to_verbose = config["paths"]["pth_to_nuc_viz"]


    Lambda = config["staining"]["lambda"] 
    poids = tuple(config["staining"]["poids"]) 

    # parse arguments
    args = parse_arguments()
    verbose = args.verbose
    slide_name = args.slide_name.split(os.path.sep)[-1] #ex : 93A_PB



    # load checkpoints
    chkpt_tumor = torch.load(f"{tumor_checkpoints}/{slide_name}_preds_probas_checkpoint.pt",weights_only=False)
    chkpt_coords = torch.load(
        f"checkpoints/coords_checkpoints/{slide_name}_coords_checkpoint.pt",weights_only=False
    )
    

    coords_x, coords_y = [], []
    for patch in os.listdir(f"{patches_path}/{slide_name}"):
        _, _, x, _, y = patch[:-4].split("_")
        coords_x.append(int(x))
        coords_y.append(int(y))

    # init pga algorithm
    Wgt = -np.log(np.array([hemato_1, eosin, safran, hemato_2]).T / 255)
    pga = PGA(Wgt, device=device)
    
    
    os.makedirs(f"data/patches_He/{slide_name}",exist_ok=True) # ex : data/patches_He/93A_PB
    # init model
    model = NucleusInstanceSegmentor(
        pretrained_model="hovernet_fast-pannuke", batch_size=256
    )

    # load tumor slides
    y_har_preds = chkpt_tumor["har_mean_preds"]
    y_harmonic = y_har_preds.numpy()
    
    # stain separation
    images = []
    for j in tqdm(range(len(y_har_preds)), desc="load images and gen Hematoxylin"):
        x, y, p = coords_x[j], coords_y[j], y_harmonic[j]
        if p != 0: # si pejorative
            try:
                # read img
                patch = f"patch_x_{x}_patch_y_{y}.jpg"
                im = plt.imread(f"{patches_path}{slide_name}/{patch}") # ex : data/patches/93A_PB/patch_x_33956_y_93057.jpg
                # stain separation
                V = vectorize(im)
                N, M, _ = im.shape
                H0 = np.maximum((pinv(Wgt) @ V), 0)

                im_He = getHstain(V, Wgt, H0, Lambda, pga, poids, n=im.shape[0])
                plt.imsave(
                    f"data/patches_He/{slide_name}/{patch.replace('.', '_He.')}", im_He
                )
                images.append(
                    f"data/patches_He/{slide_name}/{patch.replace('.', '_He.')}"
                )
            except Exception as e:
                print(f"Error in {x} {y}: {e}")
                continue
    
    # segmentation on the He patch
    images.sort()
    tile_output = model.predict(
        images,
        mode="tile",
        save_dir=save_dir + f"/{slide_name}",
        device=device,
        crash_on_exception=True,
    )

    all_x, all_y, median_areas, mean_areas, nucleocytos, densities, anisos, median_vars = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for i in tqdm(range(len(tile_output))):
        # compute nucleus features
        try:
            # read pred
            tile_preds = joblib.load(f"{tile_output[i][1]}.dat")
            # read patch
            im = plt.imread(tile_output[i][0])
            contours = get_contours_2(inst_dict=tile_preds)
            final_im = segmentNucleus(im, contours) # create a 3 colors image, white, pink, blue
            density, mean_area, median_area, aniso, _, nucyto_idx = computeFeatures(
                contours, final_im
            ) 
            if i%1000==1 and verbose:
                plt.imsave(os.path.join(path_to_verbose,'He_'+tile_output[i][0]),im)
                plt.imsave(os.path.join(path_to_verbose,'colored_'+tile_output[i][0]),final_im)
                
            _, _, x, _, y, _ = str(tile_output[i][0]).split("/")[-1].split("_")
            all_x.append(int(x))
            all_y.append(int(y))
            densities.append(density)
            mean_areas.append(mean_area)
            median_areas.append(median_area)
            anisos.append(aniso)
            nucleocytos.append(nucyto_idx)
        except Exception as e:
            print(f"Error in {str(tile_output[i][0])}: {e}")
            continue

    [x_start, y_start, _, _] = chkpt_coords["xy_start_end"]
    coords_x = np.array(coords_x) * vis_scale - x_start
    coords_y = np.array(coords_y) * vis_scale - y_start

    df_features = pd.DataFrame(
        {
            "x": all_x,
            "y": all_y,
            "density": densities,
            "median nucleus area": median_areas,
            "mean nucleus area": mean_areas,
            "anisocaryose": anisos,
            "nucleocyto index": nucleocytos,
        }
    )
    df_features.to_csv(
        f"checkpoints/nucleus_checkpoints/{slide_name}_nucleus_features.csv", index=False
    )


if __name__ == "__main__":
    main()
