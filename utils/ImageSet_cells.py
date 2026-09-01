from torch.utils.data import Dataset
from utils.Stain_Normalization import stainNorm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_tensor
from torchvision.transforms import v2
import openslide as op
import os
from utils.utils import save_data,load_data
from utils.patch_generation import mask_tissue, get_patch_coords
import yaml
from albumentations import Resize, Compose
from cellseg_models_pytorch.transforms.albu_transforms import MinMaxNormalization

class ImageSet(Dataset):
    def __init__(self, data, labels, transforms):
        self.X=data
        self.y=labels
        self.transforms=transforms

    def __len__(self):
        return len(self.y)

    def __getitem__(self,idx):
        im1,im2,im3=self.X[idx]
        imgs=[self.transforms(im) for im in [im1,im2,im3]]
        label=self.y[idx]
        return imgs,label

class ImageSet_2(Dataset):
    def __init__(self, data, transforms):
        self.X=data
        self.transforms=transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        im1,im2,im3=self.X[idx]
        imgs=[self.transforms(im) for im in [im1,im2,im3]]
        return imgs
    
class MultiscaleSet(Dataset):
    def __init__(self, slide,filtered_coords, patch_size_p,device,verbose=False,color_norm=stainNorm.DummyNormalizer()):
        self.coords = filtered_coords
        self.slide = slide
        self.patch_size_p = patch_size_p
        self.device = device
        self.verbose = verbose
        self.norm = color_norm
    def __len__(self):
        return len(self.coords)

    def __getitem__(self,idx):
        # read the patch
        center = self.coords[idx]
        x = center[0]-self.patch_size_p[0]//2
        y = center[1]-self.patch_size_p[1]//2 
        patch = np.array(self.slide.read_region((y,x),0,self.patch_size_p).convert("RGB"))
        if self.verbose:
            plt.imsave('og.png',np.array(patch))
        # normalize it
        patch = self.norm.transform(patch)
        if self.verbose:
            plt.imsave('reinhard.png',np.array(patch))
        # compute the 3 rescaled versions
        res3 = patch.shape[0]  # 1152 626 1094
        res2 = int(res3 / 1.5)
        res1 = int(res2 / 1.5)  # 512 278 486
        # find the center
        center_x, center_y = patch.shape[0] // 2, patch.shape[1] // 2
        # center crop the patch for the two smaller resolutions
        img_1 = patch[center_x - res1 // 2 : center_x + res1 // 2, center_y - res1 // 2 : center_y + res1 // 2]
        img_2 = patch[center_x - res2 // 2 : center_x + res2 // 2, center_y - res2 // 2 : center_y + res2 // 2]
        # resize the bigger resolutions to the smaller
        img_2 = (Image.fromarray(img_2)).resize((res1, res1), Image.Resampling.LANCZOS)
        img_3 = (Image.fromarray(patch)).resize((res1, res1), Image.Resampling.LANCZOS)
        # put it in a tensor 
        img_1 = pil_to_tensor(Image.fromarray(img_1)).float().to(self.device)/255 
        img_2 = pil_to_tensor(img_2).float().to(self.device)/255
        img_3 = pil_to_tensor(img_3).float().to(self.device)/255
        return img_1,img_2,img_3,x,y

class CellDetectionSet(Dataset):
    def __init__(self, slide,filtered_coords, patch_size_p,device,verbose=False,color_norm=stainNorm.DummyNormalizer()):
        self.coords = filtered_coords
        self.slide = slide
        self.patch_size_p = patch_size_p
        self.device = device
        self.verbose = verbose
        self.norm = color_norm
        self.transform = Compose([Resize(1024, 1024), MinMaxNormalization()])


    def __len__(self):
        return len(self.coords)

    def __getitem__(self,idx):
        # read the patch
        center = self.coords[idx]
        x = center[0]-self.patch_size_p[0]//2
        y = center[1]-self.patch_size_p[1]//2 
        patch = np.array(self.slide.read_region((y,x),0,self.patch_size_p).convert("RGB"))
        if self.verbose:
            plt.imsave('og.png',np.array(patch))
        # normalize it
        patch = self.norm.transform(patch)
        if self.verbose:
            plt.imsave('reinhard.png',np.array(patch))
        patch = self.transform(image=patch)['image']
        patch[patch<0]=0
        return to_tensor(patch).to(device=self.device),x,y
    
class MultiscaleSet_dummy(Dataset):
    def __init__(self, slide,filtered_coords, patch_size_p,device,ref_slide_path="data/WSIs/PB/Patient 63/63A.mrxs" ,ref_patch_path='notebooks/HES__5.jpeg', color_norm:object=stainNorm.ModifiedNormalizer(),verbose=False):
        self.coords = filtered_coords
        self.slide = slide
        self.patch_size_p = patch_size_p
        self.device = device
        self.verbose = verbose
        #requires_patch = [stainNorm_Reinhard.Normalizer().__class__, stainNorm_Reinhard.ModifiedNormalizer().__class__,
        #                   stainNorm_Reinhard.VahadaneNormalizer().__class__, stainNorm_Reinhard.MacenkoNormalizer().__class__]
        #if color_norm.__class__ == stainNorm_Reinhard.VahadaneGlobalNormalizer(slide,filtered_coords,patch_size_p).__class__:
        #elif color_norm.__class__ in requires_patch:
        #     color_norm.fit(plt.imread(ref_patch_path))
        #else:
        color_norm.fit(ref_patch_path)
        self.norm = color_norm


    def __len__(self):
        return len(self.coords)

    def __getitem__(self,idx):
        # read the patch
        center = self.coords[idx]
        x = center[0]-self.patch_size_p[0]//2
        y = center[1]-self.patch_size_p[1]//2 
        patch = np.array(self.slide.read_region((y,x),0,self.patch_size_p).convert("RGB"))
        if self.verbose:
            plt.imsave('og.png',np.array(patch))
        # normalize it
        patch = self.norm.transform(patch)
        if self.verbose:
            plt.imsave('reinhard.png',np.array(patch))
        # compute the 3 rescaled versions
        res3 = patch.shape[0]  # 1152 626 1094
        res2 = int(res3 / 1.5)
        res1 = int(res2 / 1.5)  # 512 278 486
        # find the center
        center_x, center_y = patch.shape[0] // 2, patch.shape[1] // 2
        # center crop the patch for the two smaller resolutions
        img_1 = patch[center_x - res1 // 2 : center_x + res1 // 2, center_y - res1 // 2 : center_y + res1 // 2]
        img_2 = patch[center_x - res2 // 2 : center_x + res2 // 2, center_y - res2 // 2 : center_y + res2 // 2]
        # resize the bigger resolutions to the smaller
        img_2 = (Image.fromarray(img_2)).resize((res1, res1), Image.Resampling.LANCZOS)
        img_3 = (Image.fromarray(patch)).resize((res1, res1), Image.Resampling.LANCZOS)
        # put it in a tensor 
        img_1 = pil_to_tensor(Image.fromarray(img_1)).float().to(self.device)/255 
        img_2 = pil_to_tensor(img_2).float().to(self.device)/255
        img_3 = pil_to_tensor(img_3).float().to(self.device)/255
        return img_1,img_2,img_3,x,y


       