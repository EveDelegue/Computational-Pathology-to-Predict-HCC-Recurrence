from utils.patch_generation import mask_tissue

image_path = "/home/eve/personal/CCK_survival_prediction/raw_data/3eme_batch/CKDG082.svs"
#### Patch classification (Pej vs non pej vs sain)

## extraction des patchs

# 1 masque de tissus

mask, slide = mask_tissue(image_path,verbose=True,verbose_path="brouillons/visuals_biopsy_sat4",n_threshold=4,chanel='saturation')

# 2 decoupe de coordonnées des patchs

image_path = "data/WSIs/BJ/Patient_224/224B.svs"
#### Patch classification (Pej vs non pej vs sain)

## extraction des patchs

# 1 masque de tissus

mask, slide = mask_tissue(image_path,verbose=True,verbose_path="brouillons/visuals_biopsy_sat4",n_threshold=4,chanel='saturation')

# 2 decoupe de coordonnées des patchs