import numpy as np

# read the features for each slide
# for each patient, perform a mean ?
# put the mean in a table

import os
from utils.utils_tumor import compute_class_array
from utils.utils import save_data, load_data
import yaml
import pandas
from tqdm import tqdm

# load configuration
with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
# data path and reference slide
data_pth = config["paths"]["pth_to_wsi"]
table_path = config["paths"]["pth_to_tab"]
checkpoint_path = config["paths"]["pth_to_pkl_ckpts"]

# loop through the checkpoint
patient_slides_dict = {}
slides_ckpt_list = os.listdir(checkpoint_path)
slides_ckpt_list.sort()
for slide in slides_ckpt_list:
    patient = slide[:-1]
    if patient in list(patient_slides_dict.keys()):
        patient_slides_dict[patient].append(slide)
    else:
        patient_slides_dict[patient] = [slide]

print(patient_slides_dict)
result_dict = {'Patients':[],'biggest_pej':[],'median_pej':[],
               'biggest_non_pej':[],'median_non_pej':[],
               'global_P_ratio':[], 'median_P_ratio':[], 'max_P_ratio':[]}

for patient in tqdm(list(patient_slides_dict.keys())):
    list_area_pej = []
    list_area_non_pej = []
    list_P_ratio = []
    list_class_array = []
    for slide in patient_slides_dict[patient]:
        save_path = os.path.join(checkpoint_path,slide)
        if os.path.exists(os.path.join(save_path,'area_pej.pkl')):
            list_area_pej.append(load_data(save_path,'area_pej'))
            list_area_non_pej.append(load_data(save_path,'area_non_pej'))
            list_P_ratio.append(load_data(save_path,'P_ratio'))
            list_class_array.append(compute_class_array(load_data(save_path,'tumor_dict')))
    if len(list_area_pej)>0:
        # compute biggest pej area
        result_dict['biggest_pej'].append(np.array(list_area_pej).max())
        # median pej area
        result_dict['median_pej'].append(np.median(np.array(list_area_pej)))
        # compute biggest non pej area
        result_dict['biggest_non_pej'].append(np.array(list_area_non_pej).max())
        # median non pej area
        result_dict['median_non_pej'].append(np.median(np.array(list_area_non_pej)))
        # compute global pej ratio
        global_classes = np.array(list_class_array).sum(0)
        result_dict['global_P_ratio'].append(global_classes[2]/(global_classes[2]+global_classes[1]) if (global_classes[2]+global_classes[1])>0 else 0)
        # median pej-ratio
        result_dict['median_P_ratio'].append(np.median(np.array(list_P_ratio)))
        # compute biggest non pej area
        result_dict['max_P_ratio'].append(np.array(list_P_ratio).max())
        # add to result_dict
        result_dict['Patients'].append(patient)

df = pandas.DataFrame(result_dict)
df.to_excel(os.path.join(table_path,'tumor.xlsx'))
 