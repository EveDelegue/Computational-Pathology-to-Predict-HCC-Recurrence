import numpy as np
import openslide as op
import matplotlib.pyplot as plt
import os
import cv2

def mask_tissue(tile_path:str,verbose:bool=False,verbose_path:str="brouillons/visuals")->np.ndarray:
    """detect tissue zones in the tile and return a mask
    
    :param tile_path: input tile path 
    :type tile_path: str
    :param verbose_path: path for intermediate figures. Default = "brouillons/visuals"
    :type verbose_path: str
    :param verbose: if we wish to show intermediate plots. Default = False 
    :type verbose: bool"""

    # read the slide with openslide
    slide = op.OpenSlide(tile_path)
    # get low resolution dimensions
    thumb_dimensions = slide.level_dimensions[-1]
    # read thumbnail
    thumbnail = np.asarray(slide.get_thumbnail(size=thumb_dimensions))
    if verbose:
        # visualize thumbnail
        plt.imsave(os.path.join(verbose_path,f"{os.path.basename(tile_path).split('.')[0]}_thumbnail.png"),thumbnail)
    
    # convert thumbnail to hsv 
    thumb_sat = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)[:,:,1]
    if verbose:
        # visualize thumbnail
        plt.imsave(os.path.join(verbose_path,f"{os.path.basename(tile_path).split('.')[0]}_thumbnail_sat.png"),thumb_sat)
    
    # otsu threshold 
    _, thumb_thresh = cv2.threshold(thumb_sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if verbose:
        # visualize thumbnail
        plt.imsave(os.path.join(verbose_path,f"{os.path.basename(tile_path).split('.')[0]}_thumbnail_threshold.png"),thumb_thresh)
