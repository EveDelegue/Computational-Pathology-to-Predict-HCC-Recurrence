
import matplotlib.pyplot as plt

# read the features for each slide
# for each patient, perform a mean ?
# put the mean in a table

import os
from utils.patch_generation import mask_from_xml,pad_masks
from utils.utils import load_data
from utils.utils_tumor import mask_from_tumor_dict_2
import yaml
import pandas
from tqdm import tqdm
from sklearn import metrics
import numpy as np

# load configuration
with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
# data path and reference slide
checkpoint_path = config["paths"]["pth_to_pkl_ckpts"]
seg_path = config["paths"]["pth_to_seg"]

# loop through the segmentation dataset
list_segs = [os.path.join(seg_path,path) for path in os.listdir(seg_path)]
for seg in list_segs:
    save_path = os.path.join(checkpoint_path,os.path.basename(seg).split('_')[0])
    print(os.path.join(save_path,'tumor_dict_2.pkl' ))
    # if we find a match between a seg name and a checkpoints tumor_dict_2.pkl
    if os.path.exists(os.path.join(save_path,'tumor_dict_2.pkl' )):
        # load patch size
        patch_size_p = load_data(save_path,'patch_size_p')
        # load tumor_dict
        tumor_dict_2 = load_data(save_path,'tumor_dict_2')
        # make a mask

        # parse the xml to get a mask
        mask_np_gt,mask_p_gt,mask_nt_gt = mask_from_xml(seg,max(patch_size_p[0],patch_size_p[1]))
        mask_np,mask_p,mask_nt = mask_from_tumor_dict_2(tumor_dict_2)
        # add the ground truth label to the dict entry

        tumor_dict_3 = add_gt_to_tumor_dict(tumor_dict_2)
        # compare the gt with the prediction (IoU, F1, Dice)
        mask_ref,mask_pred = pad_masks(mask_np_gt.astype(bool),mask_np.astype(bool))
        f1 = metrics.f1_score(mask_ref.flatten(),mask_pred.flatten())
        overlap = mask_ref*mask_pred
        union = mask_ref + mask_pred # Logical OR

        IOU = overlap.sum()/float(union.sum())
        plt.imsave('ref.png',mask_ref)
        plt.imsave('pred.png',mask_pred)
        print('oui')

