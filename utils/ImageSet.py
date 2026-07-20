from torch.utils.data import Dataset
from utils.Stain_Normalization import stainNorm_Reinhard
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


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
    # TODO 
    def __init__(self, slide,filtered_coords, patch_size_p,device,ref_path="notebooks/HES__5.jpeg" , color_norm=stainNorm_Reinhard.DummyNormalizer()):
        self.coords = filtered_coords
        self.slide = slide
        color_norm.fit(plt.imread(ref_path))
        self.norm = color_norm
        self.patch_size_p = patch_size_p
        self.device = device


    def __len__(self):
        return len(self.coords)

    def __getitem__(self,idx):
        # read the patch
        center = self.coords[idx]
        x = center[0]-self.patch_size_p[0]//2
        y = center[1]-self.patch_size_p[1]//2 
        patch = np.array(self.slide.read_region((y,x),0,self.patch_size_p).convert("RGB"))
        # normalize it
        patch = self.norm.transform(patch)
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