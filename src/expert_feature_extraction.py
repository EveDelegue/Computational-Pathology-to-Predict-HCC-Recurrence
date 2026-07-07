from utils.patch_generation import mask_tissue, get_patch_coords
import os

image_path = "data/WSIs/BJ/Patient_224/224B.svs"
#### Patch classification (Pej vs non pej vs sain)

## extraction des patchs

# 1 masque de tissus

mask, slide = mask_tissue(image_path,verbose=True,verbose_path="brouillons/visuals_biopsy_sat4",n_threshold=4,chanel='saturation')

# 2 decoupe de coordonnées des patchs

get_patch_coords(slide,mask,size=(287,287),step=(1,1),verbose=True,verbose_path=os.path.join("brouillons/visuals",os.path.basename(image_path).split('.')[0]))
    

image_path = "data/WSIs/HM/Patient_124/124A.ndpi"
#### Patch classification (Pej vs non pej vs sain)

## extraction des patchs

# 1 masque de tissus

mask, slide = mask_tissue(image_path,verbose=True,verbose_path="brouillons/visuals_biopsy_sat4",n_threshold=4,chanel='saturation')

# 2 decoupe de coordonnées des patchs

get_patch_coords(slide,mask,size=(287,287),step=(1,1),verbose=True,verbose_path=os.path.join("brouillons/visuals",os.path.basename(image_path).split('.')[0]))

image_path = "data/WSIs/PB/Patient_73/73B.mrxs"
#### Patch classification (Pej vs non pej vs sain)

## extraction des patchs

# 1 masque de tissus

mask, slide = mask_tissue(image_path,verbose=True,verbose_path="brouillons/visuals_biopsy_sat4",n_threshold=4,chanel='saturation')

# 2 decoupe de coordonnées des patchs

get_patch_coords(slide,mask,size=(287,287),step=(1,1),verbose=True,verbose_path=os.path.join("brouillons/visuals",os.path.basename(image_path).split('.')[0]))