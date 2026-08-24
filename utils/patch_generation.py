import numpy as np
import openslide as op
import matplotlib.pyplot as plt
import os
import cv2
from utils.utils import get_Bright_Dark_perc
from skimage.filters import threshold_multiotsu
import warnings
import itertools

def get_patch_coords(slide:op.OpenSlide,mask:np.ndarray,size:tuple[int,int]=(280,280),step:tuple[int,int]=(1,1),verbose:bool=False,verbose_path:str="brouillons/visuals",perc_bpx:float=0.3,perc_wpx:float=0.7, bright_threshold:float=0.95, dark_threshold:float=0.25)->tuple[list[tuple[int,int]],tuple[int,int]] :
    """create a list of patch coordinates in the slide inside the mask, for the desired size and step in the desired mpp. The coordinates are not necessarily of the right size, but when rescaled to the right mpp they will.
    
    :param slide: input tile  
    :type tile_path: OpenSlide
    :param verbose_path: path for intermediate figures. Default = "brouillons/visuals"
    :type verbose_path: str
    :param verbose: if we wish to show intermediate plots. Default = False 
    :type verbose: bool
    :param mask: downscaled mask of tissue segmentation
    :type mask: ndarray
    :param size: desired patch size in micron
    :type size: tuple[int,int]
    :param step: desired step between patchs in proportion
    :type step: tuple[int,int]
    :param mpp: desired resolution in mpp (0.5 mpp for 20x and 0.25 mpp for 40x)
    :type mpp: int
    :param perc_bpx: max proportion of black pixels accepted in a patch (default = 0.3)
    :type perc_bpx: float
    :param perc_wpx: max proportion of white pixels accepted in a patch (default = 0.7)
    :type perc_wpx: float
    :param bright_threshold: The grayscale intensity threshold above which pixels are considered bright. Defaults to 0.7.
    :type bright_threshold: float
    :param bright_threshold: The grayscale intensity threshold above which pixels are considered dark. Defaults to 0.05.
    :type bright_threshold: float
    """
    thumbnail = np.asarray(slide.get_thumbnail(size=slide.level_dimensions[-1])).copy()
    
    gray_thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2GRAY)
    if verbose:
        os.makedirs(verbose_path,exist_ok=True)
        # visualize gray thumbnail
        plt.imsave(os.path.join(verbose_path,"gray_thumbnail.png"),gray_thumbnail)

    # get mpp
    mpp = float(slide.properties[op.PROPERTY_NAME_MPP_X])
    # get patch size at level 0 
    patch_size_p = size[0]//mpp,size[1]//mpp # convert from micron to pixel
    patch_size_p=(int(patch_size_p[0]),int(patch_size_p[1])) # round
    # get level 0 dimensions in pixels
    dim_0 = slide.dimensions
    # get patch coordinates for the whole image
    full_coordinates = np.arange(0,dim_0[0],patch_size_p[0]*step[0],), np.arange(0,dim_0[1],patch_size_p[1]*step[1]) # any possible patch coords in x and y
    full_coords = list(itertools.product(full_coordinates[0],full_coordinates[1])) # set product
    # get downsample factor from thumbnail
    dezoom_level = max(slide.level_dimensions[0][0]/slide.level_dimensions[-1][0], slide.level_dimensions[0][1]/slide.level_dimensions[-1][1])
    # filter patchs in mask. To avoid dezoomed patchs coords out of the frame, we round under the integer
    filtered_coords = [(int(y),int(x)) for x,y in full_coords if mask[int(np.floor(y/dezoom_level)) ,int(np.floor(x/dezoom_level))] ] # only keep if mask[at center] = 1
    # filter images with too much white or dark
    bw_filtered_coords = []
    # here we take the downsampled patch. To avoid patchs of size 0 we round above the integer 
    dezoom_size = (int(np.ceil(patch_size_p[0]/(2*dezoom_level))),int(np.ceil(patch_size_p[1]/(2*dezoom_level))))
    total_pixels = 4*dezoom_size[0]*dezoom_size[1]
    for x,y in filtered_coords:
        # observe a sub_image
        sub_img = gray_thumbnail[int(x//dezoom_level)-dezoom_size[0]:int(x//dezoom_level)+dezoom_size[0],int(y//dezoom_level)-dezoom_size[1]:int(y//dezoom_level)+dezoom_size[1]]/255
        # compute the black and white ratio in the sub image
        wpx = np.sum(sub_img>bright_threshold)/total_pixels
        bpx = np.sum(sub_img< dark_threshold)/total_pixels
        # filter out the images that are too white or too black
        if wpx < perc_wpx and bpx < perc_bpx:
            bw_filtered_coords.append((x,y))
    if verbose:
        thumbnail = np.asarray(slide.get_thumbnail(size=slide.level_dimensions[-1])).copy()
        for x,y in bw_filtered_coords:
            thumbnail[int(x//dezoom_level)-1:int(x//dezoom_level)+1,int(y//dezoom_level)-1:int(y//dezoom_level)+1] = (0,0,255)
        os.makedirs(verbose_path,exist_ok=True)
        # visualize thumbnail with patchs
        plt.imsave(os.path.join(verbose_path,"thumbnail.png"),thumbnail)
        # visualize some patchs at random

        items_coords_id = np.random.choice(len(bw_filtered_coords),4)
        items_coords = np.array(bw_filtered_coords)[items_coords_id]
        for center in items_coords:
            x = center[0]-patch_size_p[0]//2
            y = center[1]-patch_size_p[1]//2 
            patch = slide.read_region((y,x),0,patch_size_p)
            plt.imsave(os.path.join(verbose_path,f"{x}_{y}_patch.png"),np.array(patch))
    return bw_filtered_coords, patch_size_p




def mask_tissue(tile_path:str,verbose:bool=False,verbose_path:str="brouillons/visuals",n_threshold:int=4,chanel:str='saturation')->tuple[np.ndarray,op.OpenSlide]:
    """detect tissue zones in the tile and return a mask
    
    :param tile_path: input tile path 
    :type tile_path: str
    :param verbose_path: path for intermediate figures. Default = "brouillons/visuals"
    :type verbose_path: str
    :param verbose: if we wish to show intermediate plots. Default = False 
    :type verbose: bool
    :param n_threshold: number of threshold for multi-threshold otsu. Default = 4
    :type n_threshold: int
    :param chanel: chanel used for thresholding. 
    'saturation' takes S chanel in HSV decomposition, 'luminance' takes oposite of L chanel from LAB decomposition, 'grey' takes the gray conversion of the image
    :type chanel: str"""

    # read the slide with openslide
    slide = op.OpenSlide(tile_path)
    # get low resolution dimensions
    thumb_dimensions = slide.level_dimensions[-1]
    # read thumbnail
    thumbnail = np.asarray(slide.get_thumbnail(size=thumb_dimensions))
    if verbose:
        # create verbose folder
        os.makedirs(verbose_path,exist_ok=True)
        # visualize thumbnail
        plt.imsave(os.path.join(verbose_path,f"{os.path.basename(tile_path).split('.')[0]}_thumbnail.png"),thumbnail)
    
    if chanel=='grey':
        # convert thumbnail to gray
        thumb_sat = 255-cv2.cvtColor(thumbnail, cv2.COLOR_RGB2GRAY)
    elif chanel=='saturation':
        # convert thumbnail to hsv and take s chanel
        thumb_sat = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)[:,:,1]
    elif chanel=='luminance':
        # convert thumbnail to LAB and take L chanel
        thumb_sat = 255-cv2.cvtColor(thumbnail, cv2.COLOR_RGB2LAB)[:,:,0]
    else:
        warnings.warn(f'name {chanel} is not an appropriate chanel name. It should be "grey", "saturation" or "luminance". Will default to saturation.')
        # convert thumbnail to hsv and take s chanel
        thumb_sat = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)[:,:,1]

    if verbose:
        # visualize thumbnail
        plt.imsave(os.path.join(verbose_path,f"{os.path.basename(tile_path).split('.')[0]}_thumbnail_sat.png"),thumb_sat)
    
    # multi otsu threshold 
    # select classes.
    thresholds = threshold_multiotsu(thumb_sat,classes=n_threshold)
    thumb_thresh = np.digitize(thumb_sat, bins=thresholds)

    if verbose:
        # visualize thumbnail thresholded
        plt.imsave(os.path.join(verbose_path,f"{os.path.basename(tile_path).split('.')[0]}_thumbnail_threshold.png"),thumb_thresh)
        # visualize thumbnail + thresholds
        contours,_ = cv2.findContours((thumb_thresh>0).astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        blue_tissue = thumbnail.copy()
        cv2.drawContours(blue_tissue,contours,-1,(0,0,255))
        plt.imsave(os.path.join(verbose_path,f"{os.path.basename(tile_path).split('.')[0]}_thumbnail_contours.png"),blue_tissue)
    return thumb_thresh>0, slide
