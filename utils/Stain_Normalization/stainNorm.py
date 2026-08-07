"""
Normalize a patch stain to the target image using the method of:

E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley, ‘Color transfer between images’, IEEE Computer Graphics and Applications, vol. 21, no. 5, pp. 34–41, Sep. 2001.
"""

from __future__ import division

import cv2 as cv
import numpy as np
import openslide as op
import cv2
from skimage.filters import threshold_multiotsu
from sklearn.decomposition import NMF
import torch
from stainx import Reinhard, Macenko, HistogramMatching
import warnings
import tqdm
warnings.filterwarnings("ignore")

### Some functions ###

def notwhite_mask(I, thresh=0.9):
    """
    Get a binary mask where true denotes 'not white'
    :param I:
    :param thresh:
    :return:
    """
    I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
    L = I_LAB[:, :, 0] / 255.0
    return (L < thresh)

def lab_split(I):
    """
    Convert from RGB uint8 to LAB and split into channels
    :param I: uint8
    :return:
    """
    I = cv.cvtColor(I, cv.COLOR_RGB2LAB)
    I = I.astype(np.float32)
    I1, I2, I3 = cv.split(I)
    return I1, I2, I3

def merge_back(I1, I2, I3):
    """
    Take seperate LAB channels and merge back to give RGB uint8
    :param I1:
    :param I2:
    :param I3:
    :return:
    """
    I = np.clip(cv.merge((I1, I2, I3)), 0, 255).astype(np.uint8)
    return cv.cvtColor(I, cv.COLOR_LAB2RGB)

def get_mean_std(I1,I2,I3):
    """
    Get mean and standard deviation of each channel
    :param I1: uint8
    :param I2: uint8
    :param I3: uint8
    :return:
    """
    m1, sd1 = cv.meanStdDev(I1)
    m2, sd2 = cv.meanStdDev(I2)
    m3, sd3 = cv.meanStdDev(I3)
    means = m1, m2, m3
    stds = sd1, sd2, sd3
    return means, stds

### Main class ###

class Normalizer(object):
    """
    A stain normalization object based on "Color transfer between images" by Reinhard
    """

    def __init__(self):
        self.target_means = [None]*3
        self.target_stds = [None]*3

    def fit(self, target):
        I1, I2, I3 = lab_split(target)
        means, stds = get_mean_std(I1, I2, I3)
        self.target_means = means
        self.target_stds = stds

    def transform(self, I):
        I1, I2, I3 = lab_split(I)
        means, stds = get_mean_std(I1,I2,I3)
        norm1 = ((I1 - means[0]) * (self.target_stds[0] / stds[0])) + self.target_means[0]
        norm2 = ((I2 - means[1]) * (self.target_stds[1] / stds[1])) + self.target_means[1]
        norm3 = ((I3 - means[2]) * (self.target_stds[2] / stds[2])) + self.target_means[2]
        return merge_back(norm1, norm2, norm3)

class ModifiedNormalizer(object):
    """
    A stain normalization object based on "Modified Reinhard Algorithm for Color Normalization of Colorectal Cancer Histopathology Images" Roy et al 2021
    """

    def __init__(self):
        self.target_means = [None]*3
        self.target_stds = [None]*3

    def fit(self, target):
        I1, I2, I3 = lab_split(target)
        means, stds = get_mean_std(I1, I2, I3)
        self.target_means = means
        self.target_stds = stds

    def transform(self, I):
        I1, I2, I3 = lab_split(I)
        means, stds = get_mean_std(I1,I2,I3)
        q = (self.target_stds[0] -stds[0])/self.target_stds[0] 
        if q>0:
            norm1 = ((I1 - means[0]) * (1+q)) + means[0]
        else:
            norm1 = ((I1 - means[0]) * (1+0.05)) + means[0]
        norm2 = (I2 - means[1]) + self.target_means[1]
        norm3 = (I3 - means[2]) + self.target_means[2]
        return merge_back(norm1, norm2, norm3)

class GlobalNormalizer2(object):
    """
    A stain normalization object based on "Color transfer between images" by Reinhard
    """

    def __init__(self,slide:op.OpenSlide,mask:np.ndarray):
        # get the mean and std of the whole image
        thumb_dimensions = slide.level_dimensions[-1]
        thumbnail = np.asarray(slide.get_thumbnail(size=thumb_dimensions))
        mean_chanels = thumbnail[mask>0].mean(0)
        std_chanels = thumbnail[mask>0].std(0)
        self.slide_mean = mean_chanels
        self.slide_std = std_chanels

    def fit(self, target:op.OpenSlide):
        # read thumbnail
        thumb_dimensions = target.level_dimensions[-1]
        thumbnail = np.asarray(target.get_thumbnail(size=thumb_dimensions))
        # get mask
        thumb_sat = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)[:,:,1]
        thresholds = threshold_multiotsu(thumb_sat,classes=4)
        mask = np.digitize(thumb_sat, bins=thresholds)
        
        mean_chanels = np.array([221.37661575, 155.08605615, 200.69394476])
        std_chanels = thumbnail[mask>0].std(0)
        self.prod = std_chanels/self.slide_std
        self.add = -(self.slide_mean* std_chanels/self.slide_std) + mean_chanels
        

    def transform(self, I):
        norm = I * self.prod + self.add
        return np.clip(norm, 0, 255).astype(np.uint8)

class GlobalMinMaxNormalizer(object):
    """
    A stain normalization object based on "Color transfer between images" by Reinhard
    """

    def __init__(self,slide:op.OpenSlide,mask:np.ndarray):
        # get the mean and std of the whole image
        thumb_dimensions = slide.level_dimensions[-1]
        thumbnail = np.asarray(slide.get_thumbnail(size=thumb_dimensions))
        mean_chanels = thumbnail[mask>0].mean(0)
        std_chanels = thumbnail[mask>0].std(0)
        self.slide_mean = mean_chanels
        self.slide_std = std_chanels

    def fit(self, target:op.OpenSlide):
        # read thumbnail
        thumb_dimensions = target.level_dimensions[-1]
        thumbnail = np.asarray(target.get_thumbnail(size=thumb_dimensions))
        # get mask
        thumb_sat = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)[:,:,1]
        thresholds = threshold_multiotsu(thumb_sat,classes=4)
        mask = np.digitize(thumb_sat, bins=thresholds)
        
        mean_chanels = thumbnail[mask>0].mean(0)
        std_chanels = thumbnail[mask>0].std(0)
        self.prod = std_chanels/self.slide_std
        self.add = -(self.slide_mean* std_chanels/self.slide_std) + mean_chanels
        

    def transform(self, I):
        norm = I.astype(float) * self.prod + self.add
        norm = 255*(norm - norm.min()) / (norm.max() - norm.min())
        return norm.astype(np.uint8)

class GlobalNormalizer(object):
    """
    A stain normalization object based on "Color transfer between images" by Reinhard
    """

    def __init__(self,slide:op.OpenSlide,mask:np.ndarray):
        # get the mean and std of the whole image
        thumb_dimensions = slide.level_dimensions[-1]
        thumbnail = np.asarray(slide.get_thumbnail(size=thumb_dimensions))
        mean_chanels = thumbnail[mask>0].mean(0)
        std_chanels = thumbnail[mask>0].std(0)
        self.slide_mean = mean_chanels
        self.slide_std = std_chanels

    def fit(self, target:op.OpenSlide):
        # read thumbnail
        thumb_dimensions = target.level_dimensions[-1]
        thumbnail = np.asarray(target.get_thumbnail(size=thumb_dimensions))
        # get mask
        thumb_sat = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)[:,:,1]
        thresholds = threshold_multiotsu(thumb_sat,classes=4)
        mask = np.digitize(thumb_sat, bins=thresholds)
        
        mean_chanels = thumbnail[mask>0].mean(0)
        std_chanels = thumbnail[mask>0].std(0)
        self.prod = std_chanels/self.slide_std
        self.add = -(self.slide_mean* std_chanels/self.slide_std) + mean_chanels
        

    def transform(self, I):
        norm = I * self.prod + self.add
        return np.clip(norm, 0, 255).astype(np.uint8)

class GlobalMeanNormalizer(object):
    """
    A stain normalization object based on "Color transfer between images" by Reinhard
    """

    def __init__(self,slide:op.OpenSlide,mask:np.ndarray):
        # get the mean and std of the whole image
        thumb_dimensions = slide.level_dimensions[-1]
        thumbnail = np.asarray(slide.get_thumbnail(size=thumb_dimensions))
        mean_chanels = thumbnail[mask>0].mean(0)
        self.slide_mean = mean_chanels

    def fit(self, target:op.OpenSlide):
        # read thumbnail
        thumb_dimensions = target.level_dimensions[-1]
        thumbnail = np.asarray(target.get_thumbnail(size=thumb_dimensions))
        # get mask
        thumb_sat = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)[:,:,1]
        thresholds = threshold_multiotsu(thumb_sat,classes=4)
        mask = np.digitize(thumb_sat, bins=thresholds)
        
        mean_chanels = thumbnail[mask>0].mean(0)
        self.add = -self.slide_mean + mean_chanels
        

    def transform(self, I):
        norm = I  + self.add
        return np.clip(norm, 0, 255).astype(np.uint8)

class LABGlobalNormalizer(object):
    """
    A stain normalization object based on "Color transfer between images" by Reinhard
    """

    def __init__(self,slide:op.OpenSlide,mask:np.ndarray):
        # get the mean and std of the whole image
        thumb_dimensions = slide.level_dimensions[-1]
        thumbnail = np.asarray(slide.get_thumbnail(size=thumb_dimensions))
        # get LAB metrics
        Lab_thumb = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2LAB)
        mean_chanels = Lab_thumb[mask>0].mean(0)
        std_chanels = Lab_thumb[mask>0].std(0)
        self.slide_mean = mean_chanels
        self.slide_std = std_chanels

    def fit(self, target:op.OpenSlide):
        # read thumbnail
        thumb_dimensions = target.level_dimensions[-1]
        thumbnail = np.asarray(target.get_thumbnail(size=thumb_dimensions))
        # get mask
        thumb_sat = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)[:,:,1]
        thresholds = threshold_multiotsu(thumb_sat,classes=4)
        mask = np.digitize(thumb_sat, bins=thresholds)
        # get LAB metrics
        Lab_thumb = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2LAB)
        mean_chanels = Lab_thumb[mask>0].mean(0)
        std_chanels = Lab_thumb[mask>0].std(0)
        self.prod = std_chanels/self.slide_std
        self.add = -(self.slide_mean* std_chanels/self.slide_std) + mean_chanels
        

    def transform(self, I):
        I_lab = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
        norm_lab = (I_lab * self.prod + self.add)
        norm_lab = np.clip(norm_lab, 0, 255).astype(np.uint8)
        norm = cv2.cvtColor(norm_lab, cv2.COLOR_LAB2RGB)
        return np.clip(norm, 0, 255).astype(np.uint8)

class ZeroOneNormalizer(object):
    """
    A stain normalization object based on "Color transfer between images" by Reinhard
    """

    def fit(self):
        pass

    def transform(self, I:np.ndarray):
        I = I.astype(float)
        norm = 255*(I - I.min()) / (I.max() - I.min())
        return norm.astype(np.uint8)

class LABGlobalMeanNormalizer(object):
    """
    A stain normalization object based on "Color transfer between images" by Reinhard
    """

    def __init__(self,slide:op.OpenSlide,mask:np.ndarray):
        # get the mean and std of the whole image
        thumb_dimensions = slide.level_dimensions[-1]
        thumbnail = np.asarray(slide.get_thumbnail(size=thumb_dimensions))
        # get LAB metrics
        Lab_thumb = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2LAB)
        mean_chanels = Lab_thumb[mask>0].mean(0)
        self.slide_mean = mean_chanels

    def fit(self, target:op.OpenSlide):
        # read thumbnail
        thumb_dimensions = target.level_dimensions[-1]
        thumbnail = np.asarray(target.get_thumbnail(size=thumb_dimensions))
        # get mask
        thumb_sat = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)[:,:,1]
        thresholds = threshold_multiotsu(thumb_sat,classes=4)
        mask = np.digitize(thumb_sat, bins=thresholds)
        # get LAB metrics
        Lab_thumb = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2LAB)
        mean_chanels = Lab_thumb[mask>0].mean(0)
        self.add = -self.slide_mean + mean_chanels
        

    def transform(self, I):
        I_lab = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
        norm_lab = (I_lab +self.add)
        norm_lab = np.clip(norm_lab, 0, 255).astype(np.uint8)
        norm = cv2.cvtColor(norm_lab, cv2.COLOR_LAB2RGB)
        return np.clip(norm, 0, 255).astype(np.uint8)

class DummyNormalizer(object):
    """
    A dummy stain normalization object
    """

    def fit(self, target):
        pass

    def transform(self, I):
        return I

def get_W_H(patch:np.ndarray):
    ## fit vahadane on target image
    mask = notwhite_mask(patch)
    # first convert RGB to OD
    OD_target = -np.log(np.clip(patch,1,255)/255)
    OD_target = OD_target[mask]
    # fit sparse NMF to get the stain matrix
    model = NMF(n_components=2,alpha_W=0.1,l1_ratio=1,alpha_H=0,init='nndsvd')
    H = model.fit_transform(OD_target)
    W = model.components_
    norm_W = np.linalg.norm(W,axis=1,keepdims=True)
    W = W/norm_W
    H = H*norm_W.T
    return W,H

def get_slide_W_Hrm(slide:op.OpenSlide,filtered_coords:list[tuple[int,int]],patch_size_p:tuple[int,int], n_sample:int=20):
    items_coords_id = np.random.choice(len(filtered_coords),n_sample) #n_sample = 20 choosen to be similar to Vahadane original paper
    items_coords = np.array(filtered_coords)[items_coords_id]
    list_W = []
    list_H = []
    for center in tqdm.tqdm(items_coords):
        x = center[0]-patch_size_p[0]//2
        y = center[1]-patch_size_p[1]//2 
        patch = np.array(slide.read_region((y,x),0,patch_size_p).convert("RGB")) #N,M,3 
        W, _ = get_W_H(patch)#2,3
        sorted_W = sorted(W,key=lambda x:x[-1])#2,3 # sort by blue intensity
        list_W.append(sorted_W)
    W_median = np.median(np.array(list_W),axis=0)
    for center in items_coords:
            x = center[0]-patch_size_p[0]//2
            y = center[1]-patch_size_p[1]//2 
            patch = np.array(slide.read_region((y,x),0,patch_size_p).convert("RGB")).reshape((-1,3))
            V = -np.log(np.clip(patch,1,255)/255)
            H = V @ np.linalg.pinv(W_median)
            H[H<0]=0
            list_H.append(H)
    H_rm = np.percentile(np.array(list_H), 99, axis=(0,1))
    return W_median, H_rm

class VahadaneGlobalNormalizer(DummyNormalizer):
    """
    A Vahadane stain normalization object
    """
    def __init__(self,slide,filtered_coords,patch_size_p) -> None:
        super().__init__()
        W, _ = get_slide_W_Hrm(slide,filtered_coords,patch_size_p)
        self.source_W_inv = np.linalg.pinv(W)
        


    def fit(self, slide, filtered_coords, patch_size_p):
        ## fit vahadane on target image
        self.target_W, self.H_rm = get_slide_W_Hrm(slide,filtered_coords,patch_size_p)

        
    def transform(self, I):
        ## fit vahadane on target image
        I_shape = I.shape
        V = -np.log(np.clip(I,1,255)/255).reshape((-1,3))
        H = V @ self.source_W_inv 
        H[H<0] = 0 
        # robust pseudo maximum of H rows
        H_rm = np.percentile(H, 99, axis=0)
        
        # norm 
        H_norm = H * self.H_rm/H_rm
        V_norm = H_norm @ self.target_W
        I_norm = np.exp(-V_norm) * 255
        return I_norm.astype(np.uint8).reshape(I_shape)


class VahadaneGlobalNormalizerW(DummyNormalizer):
    """
    A Vahadane stain normalization object
    """
    def __init__(self,W,H_rm) -> None:
        super().__init__()
        self.source_W_inv = np.linalg.pinv(W)
        self.H_rm = H_rm
        self.shape = None

    def fit(self, ref_W,ref_H_rm):
        ## fit vahadane on target image
        self.target_W, self.H_rm = ref_W, ref_H_rm

        
    def transform(self, I):
        ## fit vahadane on target image
        if self.shape==None:
            self.shape = I.shape

        V = -np.log(np.clip(I,1,255)/255).reshape((-1,3))
        H = V @ self.source_W_inv 
        H[H<0] = 0
        # robust pseudo maximum of H rows
        H_rm = np.percentile(H, 99, axis=0)
        
        # norm 
        H_norm = H * self.H_rm/H_rm
        V_norm = H_norm @ self.target_W
        I_norm = np.exp(-V_norm) * 255
        return I_norm.astype(np.uint8).reshape(self.shape)



class VahadaneNormalizer(DummyNormalizer):
    """
    A Vahadane stain normalization object
    """

    def fit(self, target):
        ## fit vahadane on target image
        mask = notwhite_mask(target)
        # first convert RGB to OD
        OD_target = -np.log(np.clip(target,1,255)/255)
        OD_target = OD_target[mask]
        # fit sparse NMF to get the stain matrix
        model = NMF(n_components=2,alpha_W=0.1,l1_ratio=1,alpha_H=0,init='nndsvd')
        H = model.fit_transform(OD_target)
        self.W = model.components_
        self.H_rm = np.percentile(H, 99, axis=0)

        
    def transform(self, I):
        ## fit vahadane on target image
        mask = notwhite_mask(I)
        # first convert RGB to OD
        OD = -np.log(np.clip(I,1,255)/255)
        OD = OD[mask]
        # fit sparse NMF to get the stain matrix
        model = NMF(n_components=2,alpha_W=0.1,l1_ratio=1,alpha_H=0,init='nndsvd')
        H = model.fit_transform(OD)
        W = model.components_
        # compare W with self.W and reorder if necessary
        if (W[0]*self.W[0]).sum()/(np.linalg.norm(W[0])*np.linalg.norm(self.W[0])) < (W[0]*self.W[1]).sum()/(np.linalg.norm(W[0])*np.linalg.norm(self.W[1])): 
            W = W[::-1]
            H = H[:,::-1]
        # robust pseudo maximum of H rows
        H_rm = np.percentile(H, 99, axis=0)

        # norm 
        H_norm = H * self.H_rm/H_rm
        V_norm = H_norm @ self.W
        I_norm = np.exp(-V_norm) * 255
        norm = I.copy()
        norm[mask] = I_norm
        return norm.astype(np.uint8)

class MacenkoNormalizer(object):
    """
    A dummy stain normalization object
    """
    def __init__(self):
        self.normalizer = Macenko()

    def fit(self, target):
        torch_target = torch.as_tensor(target).permute(2,0,1).unsqueeze(0)
        self.normalizer.fit(torch_target)

    def transform(self, I):
        torch_I = torch.as_tensor(I).permute(2,0,1).unsqueeze(0)
        return self.normalizer.transform(torch_I).squeeze().permute(1,2,0).cpu().numpy()

