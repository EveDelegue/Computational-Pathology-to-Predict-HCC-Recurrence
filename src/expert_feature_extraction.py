from utils.patch_generation import mask_tissue

image_path = "/home/eve/personal/papier_aymen/Computational-Pathology-to-Predict-HCC-Recurrence/data/WSIs/BJ/Patient_224/224B.svs"
#### Patch classification (Pej vs non pej vs sain)

## extraction des patchs

# 1 masque de tissus

mask_tissue(image_path,verbose=True)

image_path = "/home/eve/personal/papier_aymen/Computational-Pathology-to-Predict-HCC-Recurrence/data/WSIs/PB/Patient_73/73B.mrxs"
#### Patch classification (Pej vs non pej vs sain)

## extraction des patchs

# 1 masque de tissus

mask_tissue(image_path,verbose=True)

image_path = "/home/eve/personal/papier_aymen/Computational-Pathology-to-Predict-HCC-Recurrence/data/WSIs/HM/Patient_124/124A.ndpi"
#### Patch classification (Pej vs non pej vs sain)

## extraction des patchs

# 1 masque de tissus

mask_tissue(image_path,verbose=True)