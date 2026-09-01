from utils.utils_tumor import mask_tumor
import os
from utils.utils import save_data, load_data
import openslide as op
import yaml

# load configuration
with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
# data path and reference slide
data_pth = config["paths"]["pth_to_wsi"]
ref_slide_pth = config["staining"]["ref_slide"]
ref_save_path = os.path.join(config["paths"]["pth_to_pkl_ckpts"],os.path.basename(ref_slide_pth).split('.')[0])

hospital_list = os.listdir(data_pth)
patients_list = []
for hospital in hospital_list:
        patients_list.extend([os.path.join(data_pth,hospital,patient) for patient in os.listdir(os.path.join(data_pth,hospital))])

slides_list = [ref_slide_pth] # process ref slide first
for patient in patients_list:
        slides_list.extend([os.path.join(patient,slide) for slide in os.listdir(patient) if ((('.mrxs' in slide) or ('.ndpi' in slide)) and (slide!=os.path.basename(ref_slide_pth)))])

# loop through the slides
for image_path in slides_list:
    # init save paths
    save_path = os.path.join(config["paths"]["pth_to_pkl_ckpts"],os.path.basename(image_path).split('.')[0])
    visual_path = os.path.join(config["paths"]["pth_to_verbose"],os.path.basename(image_path).split('.')[0])

    tumor_dict = load_data(save_path,'tumor_dict')
    patch_size_p = load_data(save_path,'patch_size_p')
    slide = op.OpenSlide(image_path)



    # 4 create tumor mask
    if not(os.path.exists(os.path.join(save_path,'in_mask.pkl')) and os.path.exists(os.path.join(save_path,'area_pej.pkl'))):
            # if not done yet
            in_mask,out_mask, P_ratio, area_pej, area_non_pej, tumor_dict_2 = mask_tumor(tumor_dict,patch_size_p,slide=slide,verbose=True,verbose_path=visual_path)
            save_data(save_path,{'in_mask':in_mask, 'out_mask':out_mask, 'P_ratio':P_ratio, 'area_pej':area_pej, 'area_non_pej': area_non_pej, 'tumor_dict_2':tumor_dict_2})
    else:
            # else load it
            in_mask = load_data(save_path,'in_mask')
            out_mask = load_data(save_path,'out_mask')
            tumor_dict_2 = load_data(save_path,'tumor_dict_2')

    # 5 sample patchs for cell analysis
    #if not(os.path.exists(os.path.join(save_path,'sampled_patchs.pkl'))):
    #      sampled_patchs = sample_patchs(in_mask,1/10)
          
    
   