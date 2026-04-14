import torch
import os
import yaml
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

from utils.utils_nucleus import computeFeaturesArea, getNucleusFeaturesArea
from utils.PGA import PGA
import time
import multiprocessing

device = "cuda" if torch.cuda.is_available() else "cpu"


def process_patch(args):
    """Process a single patch for parallel execution.
    :param args: contains all the required arguments"""
    (j,coords_x,coords_y,y_harmonic,patches_path,slide_name,Wgt,Lambda,
        poids,kernel_size,verbose,path_to_verbose,mpp,ref_mpp,device ) = args
    
    x, y, p = coords_x[j], coords_y[j], y_harmonic[j]
    if p != 0:  # si tumorale
        # read img
        patch = f"patch_x_{x}_y_{y}.jpg"
        im = plt.imread(f"{patches_path}/{slide_name}/{patch}")
        # initialize GPU/CPU object in worker process
        pga = PGA(Wgt, device=device)
        # stain separation and nuclei detection
        areas, final_im, density = getNucleusFeaturesArea(im,Wgt,Lambda,pga,poids,kernel_size,verbose,
            path_to_verbose + '/' + str(j) + '_' + str(slide_name),mpp,ref_mpp)
        # features for prediction
        density, mean_area, median_area, aniso, _, nucyto_idx = computeFeaturesArea(areas, final_im, density)
        return int(x), int(y), density, mean_area, median_area, aniso, nucyto_idx
    else:
        return None


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide_name", type=str,default='data/patches/42F_PB')
    parser.add_argument("--verbose",type=bool,default=False)
    args = parser.parse_args()
    return args

def main():

    # load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # pga params
    hemato_1 = config["staining"]["hemato_1"]
    eosin = config["staining"]["eosin"]
    safran = config["staining"]["safran"]
    hemato_2 = config["staining"]["hemato_2"]

    Lambda = config["staining"]["lambda"] 
    poids = tuple(config["staining"]["poids"]) 
    kernel_size = config["nucleus_model"]["kernel_size"]

    vis_scale = config["patching"]["vis_scale"]
    # paths
    tumor_checkpoints = config["paths"]["pth_to_tumor_ckpts"]
    patches_path = config["paths"]["pth_to_patches"]
    coords_checkpoints = config["paths"]["pth_to_coords"]
    save_dir = config["paths"]["pth_to_nuc_ckpts"]

    path_to_verbose = config["paths"]["pth_to_nuc_viz"]

    # parse arguments
    args = parse_arguments()
    verbose = args.verbose
    slide_name = args.slide_name.split(os.path.sep)[-1] #ex : 93A_PB

    hospital = slide_name.split('_')[-1] 
    mpp_dict = config["patching"]["mpp_dict"]
    if hospital not in mpp_dict.keys():
        raise KeyError(f"no resolution defined for this hospital : {hospital}. Check that the hospital's name in the config file is the same as in the data folder.")
    mpp = mpp_dict[hospital]
    ref_mpp = mpp_dict[config["patching"]["ref_hospital"]]

    # load tumor checkpoints
    chkpt_tumor = torch.load(f"{tumor_checkpoints}/{slide_name}_preds_probas_checkpoint.pt",weights_only=False)
    # load coordinates
    chkpt_coords = torch.load(
        f"{coords_checkpoints}/{slide_name}_coords_checkpoint.pt",weights_only=False
    )
    
    coords_x, coords_y = [], []
    for patch in os.listdir(f"{patches_path}/{slide_name}"):
        _, _, x, _, y = patch[:-4].split("_")
        coords_x.append(int(x))
        coords_y.append(int(y))

    # init pga algorithm
    Wgt = -np.log(np.array([hemato_1, eosin, safran, hemato_2]).T / 255)
    pga = PGA(Wgt, device=device)

    # load tumor slides
    y_har_preds = chkpt_tumor["har_mean_preds"]
    y_harmonic = y_har_preds.numpy()
    
    all_x, all_y, median_areas, mean_areas, nucleocytos, densities, anisos = (
        [],
        [],
        [],
        [],
        [],
        [],
        []
    )

    t0 = time.time()
    # stain separation v2 using parallelization
    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(multiprocessing.cpu_count()//2)

    # Prepare arguments for each patch
    patch_args = [
        (
            j,
            coords_x,
            coords_y,
            y_harmonic,
            patches_path,
            slide_name,
            Wgt,
            Lambda,
            poids,
            kernel_size,
            verbose,
            path_to_verbose,
            mpp,
            ref_mpp,
            device
        )
        for j in range(len(y_har_preds))
    ]
    # run the multiprocess 
    out1 = pool.map(process_patch, patch_args)
    # close the multiprocessing pool
    pool.close()
    # wait for it to be closed
    pool.join()
    
    # Process results
    for result in out1:
        if result is not None:
            x, y, density, mean_area, median_area, aniso, nucyto_idx = result
            all_x.append(x)
            all_y.append(y)
            densities.append(density)
            mean_areas.append(mean_area)
            median_areas.append(median_area)
            anisos.append(aniso)
            nucleocytos.append(nucyto_idx)
    t1 = time.time()
    print(f'time processing : {t1-t0}')

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
        f"{save_dir}/{slide_name}_nucleus_features.csv", index=False
    )

if __name__ == "__main__":
    main()
