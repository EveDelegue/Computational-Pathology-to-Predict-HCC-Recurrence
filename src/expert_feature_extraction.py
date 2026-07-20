from utils.patch_generation import mask_tissue, get_patch_coords
from utils.utils_tumor import detect_architectures, mask_tumor
import os
from utils.utils import save_data, load_data
import openslide as op



image_path = "data/WSIs/BJ/Patient_229/229C.svs"
save_path = os.path.join('checkpoints/pickles',os.path.basename(image_path).split('.')[0])
visual_path = os.path.join("brouillons/visuals",os.path.basename(image_path).split('.')[0])
#### Patch classification (Pej vs non pej vs sain)

## extraction des patchs

# 1 masque de tissus
if not(os.path.exists(os.path.join(save_path,'mask.pkl'))):

    mask, slide = mask_tissue(image_path,verbose=True,verbose_path=os.path.join("brouillons/visuals",os.path.basename(image_path).split('.')[0]),n_threshold=4,chanel='saturation')
    save_data(save_path,{'mask':mask})
else:
    mask = load_data(save_path,'mask')
    slide = op.OpenSlide(image_path)
# 2 decoupe de coordonnées des patchs

if not(os.path.exists(os.path.join(save_path,'filtered_coords.pkl')) and os.path.exists(os.path.join(save_path,'patch_size_p.pkl'))):
    filtered_coords,patch_size_p = get_patch_coords(slide,mask,size=(280,280),step=(1,1),verbose=True,verbose_path=os.path.join("brouillons/visuals",os.path.basename(image_path).split('.')[0]))
    save_data(save_path,{'filtered_coords':filtered_coords,'patch_size_p':patch_size_p})
else:
    filtered_coords = load_data(save_path,'filtered_coords')
    patch_size_p = load_data(save_path,'patch_size_p')

# 3 use in pej non-pej pipeline
if not(os.path.exists(os.path.join(save_path,'tumor_dict.pkl'))):
    tumor_dict = detect_architectures(slide,filtered_coords,patch_size_p,model_path="models",verbose=True,verbose_path=os.path.join("brouillons/visuals",os.path.basename(image_path).split('.')[0]))
    save_data(save_path,{'tumor_dict':tumor_dict})
else:
    tumor_dict = load_data(save_path,'tumor_dict')

# 4 create tumor mask
in_mask,out_mask = mask_tumor(tumor_dict,patch_size_p,slide=slide,verbose=True,verbose_path=os.path.join("brouillons/visuals",os.path.basename(image_path).split('.')[0]))