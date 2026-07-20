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


### Some functions ###


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



class DummyNormalizer(object):
    """
    A dummy stain normalization object
    """

    def fit(self, target):
        pass

    def transform(self, I):
        return I