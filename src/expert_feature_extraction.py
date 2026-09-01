from utils.patch_generation import mask_tissue, get_patch_coords, sample_patchs
from utils.utils_tumor import detect_architectures, mask_tumor
import os
from utils.utils import save_data, load_data
import openslide as op
import yaml
from utils.Stain_Normalization import stainNorm

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
    #### Patch classification (Pej vs non pej vs sain)

    ## extraction des patchs

    # 1 masque de tissus
    if not(os.path.exists(os.path.join(save_path,'mask.pkl'))):
        # if not done yet
        mask, slide = mask_tissue(image_path,verbose=True,verbose_path=visual_path,n_threshold=4,chanel='saturation')
        save_data(save_path,{'mask':mask})
    else:
        # else load it
        mask = load_data(save_path,'mask')

        slide = op.OpenSlide(image_path)

    # 2 decoupe de coordonnées des patchs
    if not(os.path.exists(os.path.join(save_path,'filtered_coords.pkl')) and os.path.exists(os.path.join(save_path,'patch_size_p.pkl'))):
        # if not done yet
        filtered_coords,patch_size_p = get_patch_coords(slide,mask,size=(280,280),step=(1,1),verbose=True,verbose_path=visual_path)
        save_data(save_path,{'filtered_coords':filtered_coords,'patch_size_p':patch_size_p})
    else:
        # else load it
        filtered_coords = load_data(save_path,'filtered_coords')
        patch_size_p = load_data(save_path,'patch_size_p')


    ## detection de la tumeur
    if not(os.path.exists(os.path.join(save_path,'tumor_dict.pkl'))):

        # 1 compute ref color
        if not(os.path.exists(os.path.join(ref_save_path,'W.pkl'))):
            # load ref slide data
            ref_slide =  op.OpenSlide(ref_slide_pth)
            ref_filtered_coords= load_data(ref_save_path,'filtered_coords')
            ref_patch_size_p = load_data(ref_save_path,'patch_size_p')
            # compute ref stain values
            ref_W,ref_H_rm = stainNorm.get_slide_W_Hrm(ref_slide,ref_filtered_coords,ref_patch_size_p,n_sample=100)
            save_data(ref_save_path,{'W':ref_W,'H_rm':ref_H_rm})
        else:
            # load stain values directly
            ref_W = load_data(ref_save_path,'W')
            ref_H_rm = load_data(ref_save_path,'H_rm')

        # 2 compute source color
        if not(os.path.exists(os.path.join(save_path,'W.pkl'))):
                    # get values
                    W,H_rm = stainNorm.get_slide_W_Hrm(slide,filtered_coords,patch_size_p)
                    save_data(save_path,{'W':W,'H_rm':H_rm})
        else:       
                    # or load them
                    W = load_data(save_path,'W')
                    H_rm = load_data(save_path,'H_rm')

        # 3 use in pej non-pej pipeline
        norm_dict = {'W':W,'H_rm':H_rm,"ref_W":ref_W,"ref_H_rm":ref_H_rm}
        tumor_dict = detect_architectures(slide,filtered_coords,patch_size_p,model_path=config['model']['path_triple_resnets'],verbose=True,verbose_path=visual_path,batch_size=config['model']['batch_size'],norm_dict=norm_dict)
        save_data(save_path,{'tumor_dict':tumor_dict})
    else:
        # else load results
        tumor_dict = load_data(save_path,'tumor_dict')
        W = load_data(save_path,'W')
        H_rm = load_data(save_path,'H_rm')

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
          
    
   