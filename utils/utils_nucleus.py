import cv2
import torch
import numpy as np
from numpy.linalg import pinv
from skimage import filters
import matplotlib.pyplot as plt
import os
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
import skimage
from skimage.segmentation import watershed

from skimage.feature import peak_local_max
from scipy import ndimage

def vectorize(im, N=500 * 500):
    N, M, _ = im.shape
    N *= M
    Im = -np.log(np.where(im == 0, 1, im) / 255)
    V = np.zeros((3, N))
    V[0, :] = Im[:, :, 0].flatten()
    V[1, :] = Im[:, :, 1].flatten()
    V[2, :] = Im[:, :, 2].flatten()
    return V


def unvectorize(V, N=500):
    return np.exp(-V.reshape((3, N, N)).transpose([1, 2, 0]))


def getStainsBis(W, H, poids=[1.0, 1.0], n=500):
    c1 = np.kron(W[:, 0, np.newaxis], np.transpose(H[0, :, np.newaxis]))
    c4 = np.kron(W[:, 3, np.newaxis], np.transpose(H[3, :, np.newaxis]))
    c1 = poids[0] * c1 + poids[1] * c4
    im_c1 = np.zeros((n, n, 3))
    im_c1[:, :, 0], im_c1[:, :, 1], im_c1[:, :, 2] = (
        c1[0, :].reshape(n, n),
        c1[1, :].reshape(n, n),
        c1[2, :].reshape(n, n),
    )
    im_c1 = np.exp(-im_c1)
    c2 = np.kron(W[:, 1, np.newaxis], np.transpose(H[1, :, np.newaxis]))
    im_c2 = np.zeros((n, n, 3))
    im_c2[:, :, 0], im_c2[:, :, 1], im_c2[:, :, 2] = (
        c2[0, :].reshape(n, n),
        c2[1, :].reshape(n, n),
        c2[2, :].reshape(n, n),
    )
    im_c2 = np.exp(-im_c2)
    c3 = np.kron(W[:, 2, np.newaxis], np.transpose(H[2, :, np.newaxis]))
    im_c3 = np.zeros((n, n, 3))
    im_c3[:, :, 0], im_c3[:, :, 1], im_c3[:, :, 2] = (
        c3[0, :].reshape(n, n),
        c3[1, :].reshape(n, n),
        c3[2, :].reshape(n, n),
    )
    im_c3 = np.exp(-im_c3)
    return im_c1, im_c2, im_c3


def gen_HES(W, H_rec, N, M, device, BS):
    c11 = torch.matmul(W[:, 0].unsqueeze(1), H_rec[:, 0, :].unsqueeze(1))
    c12 = torch.matmul(W[:, 3].unsqueeze(1), H_rec[:, 3, :].unsqueeze(1))
    im_c1 = torch.exp(-(c11 + c12).reshape(BS, 3, N, M)).to(device)
    c2 = torch.matmul(W[:, 1].unsqueeze(1), H_rec[:, 1, :].unsqueeze(1))
    im_c2 = torch.exp(-c2.reshape(BS, 3, N, M)).to(device)
    c3 = torch.matmul(W[:, 2].unsqueeze(1), H_rec[:, 2, :].unsqueeze(1))
    im_c3 = torch.exp(-c3.reshape(BS, 3, N, M)).to(device)
    return im_c1, im_c2, im_c3


def getHstain(V, W, H0, Lambda, model, poids, n=512):
    H_rec = model(V, H0, Lambda)
    im_H, _, _ = getStainsBis(W, H_rec, poids, n)
    return (255 * im_H).astype(np.uint8)


def getNucleusMask(im_He,gaussian_filter=(31,31)):
    blur_c1 = cv2.GaussianBlur(
        cv2.cvtColor(im_He, cv2.COLOR_RGB2GRAY), gaussian_filter, 0
    )
    #blur_c1 = cv2.cvtColor(im_He, cv2.COLOR_RGB2GRAY)
    thresholds = filters.threshold_multiotsu(blur_c1, classes=3)
    multiotsu_mask = np.invert(255 * (blur_c1 > thresholds[0]).astype(np.uint8))
    return multiotsu_mask


def getCleanMask(mask, scaled_kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(scaled_kernel_size,scaled_kernel_size))
    opened_mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN, kernel)
    return opened_mask

def get_contours(
    inst_dict: dict, inst_colours: np.ndarray | tuple[int, int, int] = (255, 255, 0)
):
    if inst_colours is None:
        inst_colours = random_colors(len(inst_dict), bright=True)

    if not isinstance(inst_colours, (tuple, np.ndarray)):
        msg = f"`inst_colours` must be np.ndarray or tuple: {type(inst_colours)}."
        raise TypeError(
            msg,
        )

    inst_colours_array = np.array(inst_colours) * 255 # problème de normalisation ici (255 pris deux fois)

    if isinstance(inst_colours, tuple):
        inst_colours_array = np.array([inst_colours] * len(inst_dict))

    inst_colours_array = inst_colours_array.astype(np.uint8)
    contours = []
    for _, [_, inst_info] in enumerate(inst_dict.items()):
        inst_contour = inst_info["contour"]
        contours.append(np.array(inst_contour))
    return contours

def get_contours_2(inst_dict: dict):
    
    contours = []
    for _, [_, inst_info] in enumerate(inst_dict.items()):
        inst_contour = inst_info["contour"]
        contours.append(np.array(inst_contour))
    return contours


def detectContours(im, opened_mask):
    contours, _ = cv2.findContours(
        opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )  # Find contours in the binary image
    convex_contours = contours#[]
    #for cnt in contours:
    #    convex_cnt = cv2.convexHull(cnt)
    #    convex_contours.append(convex_cnt)
    contour_im = cv2.drawContours(im.copy(), convex_contours, -1, (0, 0, 0), thickness=2)
    return contour_im, convex_contours

def detectEllipsisContours(im,edges,accuracy=20, threshold=90, min_size=15, max_size=40):
    result = hough_ellipse(edges,accuracy,threshold,min_size,max_size)
    # Estimated parameters for the ellipse
    contours = []
    im_contours = im.copy()
    for elipsis in result:
        acc,yc, xc, a, b = [int(round(x)) for x in list(elipsis)[:5]]
        if acc> threshold and a>0 and b>0:
            orientation = elipsis[5]
            # Draw the ellipse on the original image
            cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
            contours.append(np.stack([[cx],[cy]]).T)
            im_contours[cy,cx] = (0,250,0)
    return im_contours, contours
    
    
    

def detectNucleus(contour_im, contours,inf_p=35,inf_a=35): #inf_p=35, inf_a=35):
    perimeters, areas = [], []
    filtred_contours = []
    for contour in contours:
        a, p = cv2.contourArea(contour), cv2.arcLength(contour, True)
        perimeters.append(p)
        areas.append(a)
        if p > inf_p and a > inf_a:
            r = p/(2*np.pi)
            if np.pi*(r**2)<2*a:  
                filtred_contours.append(contour)
    # Draw the detected contours on the mask
    contour_im0 = cv2.drawContours(
        contour_im.copy(), filtred_contours, -1, (0, 255, 0), thickness=2
    )
    return contour_im0, filtred_contours


def segmentNucleus(
    im,
    filtred_contours,
    lw=np.array([200, 200, 200], dtype=np.uint8),
    uw=np.array([255, 255, 255], dtype=np.uint8),
):
    # Create a mask to identify white pixels within the specified range
    white_mask = cv2.inRange(im, lw, uw)
    final_im = im.copy()
    final_im[white_mask > 0] = [255, 255, 255]  # type: ignore
    final_im = cv2.drawContours(
        final_im, filtred_contours, -1, (0, 0, 128), cv2.FILLED
    )  # Draw the detected contours on the mask
    # Combine the two masks to identify pixels that are either color1 or color2
    combined_mask = np.logical_or(
        np.all(final_im == (255, 255, 255), axis=-1),
        np.all(final_im == (0, 0, 128), axis=-1),
    )
    final_im[~combined_mask] = (199, 21, 133)
    return final_im


def computeFeatures(filtred_contours, final_im):
    density = len(filtred_contours)
    areas = [cv2.contourArea(contour) for contour in filtred_contours]
    mean_area = np.mean(areas)
    median_area = np.median(areas)
    anisocaryose = np.std(areas)
    median_variance_area = np.median([np.abs(area - median_area) for area in areas])
    if np.sum(np.all(final_im == (199, 21, 133), axis=-1)) == 0:
        nucleocyto_idx = 0
    else:
        blue = np.sum(np.all(final_im == (0, 0, 128), axis=-1))
        pink = np.sum(np.all(final_im == (199, 21, 133), axis=-1)) + 1e-9
        nucleocyto_idx = blue / pink
    return (
        density,
        mean_area,
        median_area,
        anisocaryose,
        median_variance_area,
        nucleocyto_idx,
    )

def computeFeaturesArea(areas, final_im,density):
    mean_area = np.mean(areas)
    median_area = np.median(areas)
    anisocaryose = np.std(areas)
    median_variance_area = np.median([np.abs(area - median_area) for area in areas])
    if np.sum(np.all(final_im == (199, 21, 133), axis=-1)) == 0:
        nucleocyto_idx = 0
    else:
        blue = np.sum(np.all(final_im == (0, 0, 128), axis=-1))
        pink = np.sum(np.all(final_im == (199, 21, 133), axis=-1)) + 1e-9
        nucleocyto_idx = blue / pink
    return (
        density,
        mean_area,
        median_area,
        anisocaryose,
        median_variance_area,
        nucleocyto_idx,
    )

def getWatershed(clean_mask:np.ndarray,footprint:int=10)->np.ndarray:
    """Performs watershed to separate neighborhoods of nuclei
    
    :param clean_mask: input mask
    :type clean_mask: np.ndarray
    :param footprint: min distance between two nuclei center
    :type footprint: int"""
    # ensure type is uint8
    image = clean_mask.astype(np.uint8)
    # compute distance transform
    distance= cv2.distanceTransform(image,cv2.DIST_L2, 3)
    # find max of the local max of the distance transform
    max_coords = peak_local_max(distance, labels=image,
                                footprint=np.ones((footprint, footprint)))
    # put it in a map
    local_maxima = np.zeros_like(image, dtype=bool)
    local_maxima[tuple(max_coords.T)] = True
    # label it
    markers = ndimage.label(local_maxima)[0]

    # apply watershed
    output = watershed(-distance, markers, mask=image)
    return output

def getAreas(im:np.ndarray,watershed_im: np.ndarray,min_area:float = 35,lw:int=200,
    uw:int=255,ratio:float=1)->tuple[list,np.ndarray,int]:
    """Compute nucleas areas, pink and blue image, and number of nucleus.
    
    :param im: HES input image
    :type im: np.ndarray
    :param watershed_im: watershed nucleus separation of the input image
    :type watershed_im: np.ndarray
    :param min_area: minimum area accepted for a nucleus
    :type min_area: float
    :param lw: lower bound of the background intensity
    :type lw: int
    :param uw: upper bound of the background intensity
    :type uw: int
    :param ratio: Surface of the ref pixel compared to actual pixel (surface PB = 1). Usually ratio = (ref_mpp/mpp)**2
    :type ratio: float"""

    lw_rgb=np.array([lw,lw, lw], dtype=np.uint8)
    uw_rgb=np.array([uw,uw, uw], dtype=np.uint8)
    # Create a mask to identify white pixels within the specified range
    white_mask = cv2.inRange(im, lowerb=lw_rgb, upperb=uw_rgb)
    # create the final image
    final_im = im.copy()
    # background is white
    final_im[white_mask > 0] = [255, 255, 255]

    # list all nuclei
    areas = []
    for nucleus_id in range(1,watershed_im.max()+1):
        area = np.sum(watershed_im==nucleus_id)/ratio
        # filter the small ones
        if area>min_area:
            areas.append(area)
            # nuclei is blue
            final_im[watershed_im==nucleus_id] = (0, 0, 128)

    combined_mask = np.logical_or(
        np.all(final_im == (255, 255, 255), axis=-1),
        np.all(final_im == (0, 0, 128), axis=-1),
    )
    # the rest is pink
    final_im[~combined_mask] = (199, 21, 133)

    # number of nuclei for each patch
    density = len(areas)
    return areas,final_im,density

def getNucleusFeatures(im, W, Lambda, model, poids, kernel_size = 5,verbose=False,verbose_path='',mpp=0.25,ref_mpp=0.25):
    '''Extract nucleus contours from a patch.
    im : a plt.imread image
    W : -np.log(np.array([hemato_1, eosin, safran, hemato_2]).T / 255) an array containing the staining reference
    model, Lambda, poids : pga model and its parameters
    kernel_size : int, diameter of the morphological structuring element
    mpp : patch resolution
    ref_mpp : ref center (usually PB) patch resolution
    verbose : if True save intermediate images
    verbose_path : path to save these images'''

    scaled_kernel_size = 2 * int((kernel_size*ref_mpp/mpp)/2) + 1 # must be odd and the closest to kernel_size*ref_mpp/mpp

    V  = vectorize(im)
    im_He = getHstain(V, W, np.maximum((pinv(W) @ V), 0), Lambda, model, poids,n=im.shape[0]) 
    mask = getNucleusMask(im_He)
    clean_mask = getCleanMask(mask, scaled_kernel_size)
    contour_im0, contours = detectContours(im, clean_mask)
    contours_2 , filtred_contours = detectNucleus(contour_im0, contours)
    final_im = segmentNucleus(im, filtred_contours)
    if verbose: 
        # save intermediate images for debugging
        os.makedirs(verbose_path,exist_ok=True)
        plt.imsave(f"{verbose_path}/im.png", im)
        plt.imsave(f"{verbose_path}/im_He.png", im_He)
        plt.imsave(f"{verbose_path}/mask.png", mask, cmap='gray')
        plt.imsave(f"{verbose_path}/clean_mask.png", clean_mask, cmap='gray')
        plt.imsave(f"{verbose_path}/contour_im0.png", contour_im0)
        plt.imsave(f"{verbose_path}/contours_2.png", contours_2)
        plt.imsave(f"{verbose_path}/final_im.png", final_im)
    return final_im, filtred_contours

def getNucleusFeaturesArea(im, W, Lambda, model, poids, kernel_size = 5,verbose=False,verbose_path='',mpp=0.25,ref_mpp=0.25,
                           footprint=10,min_area=100,gauss_kernel=31):
    '''Extract nucleus contours from a patch.
    im : a plt.imread image
    W : -np.log(np.array([hemato_1, eosin, safran, hemato_2]).T / 255) an array containing the staining reference
    model, Lambda, poids : pga model and its parameters
    kernel_size : int, diameter of the morphological structuring element
    mpp : patch resolution
    ref_mpp : ref center (usually PB) patch resolution
    verbose : if True save intermediate images
    verbose_path : path to save these images'''

    # scale the sizes to fit different mmps
    scaled_kernel_size = 2 * int((kernel_size*ref_mpp/mpp)/2) + 1 # must be odd and the closest to kernel_size*ref_mpp/mpp
    scaled_footprint = 2 * int((footprint*ref_mpp/mpp)/2) + 1 # must be odd and the closest to footprint*ref_mpp/mpp
    scaled_gauss = 2 * int((gauss_kernel*ref_mpp/mpp)/2) + 1 # must be odd and the closest to gauss_kernel*ref_mpp/mpp

    V  = vectorize(im)
    im_He = getHstain(V, W, np.maximum((pinv(W) @ V), 0), Lambda, model, poids,n=im.shape[0]) 
    mask = getNucleusMask(im_He,gaussian_filter=(scaled_gauss,scaled_gauss))
    clean_mask = getCleanMask(mask, scaled_kernel_size)
    watershed_image = getWatershed(clean_mask,scaled_footprint)
    areas,final_im,density = getAreas(im,watershed_image,min_area,ratio=(ref_mpp/mpp)**2)
    if verbose: 
        # save intermediate images for debugging
        os.makedirs(verbose_path,exist_ok=True)
        plt.imsave(f"{verbose_path}/im.png", im)
        plt.imsave(f"{verbose_path}/im_He.png", im_He)
        plt.imsave(f"{verbose_path}/mask.png", mask, cmap='gray')
        plt.imsave(f"{verbose_path}/clean_mask.png", clean_mask, cmap='gray')
        plt.imsave(f"{verbose_path}/watershed.png", watershed_image, cmap='gist_ncar')
        water_on_im = im.copy()
        water_on_im[final_im==(0,0,128)]=final_im[final_im==(0,0,128)]
        plt.imsave(f"{verbose_path}/watershed_viz.png", water_on_im)
        plt.imsave(f"{verbose_path}/final_im.png", final_im)
    return areas,final_im,density

def getEdges(im_He,g_kernel_size):
    #grayscale = cv2.cvtColor(im_He, cv2.COLOR_RGB2GRAY)
    canny = skimage.feature.canny(im_He,sigma=g_kernel_size)
    return canny

def getNucleusFeatures_2(im,  W, Lambda, model, poids, g_kernel_size = 7, verbose=False,verbose_path='',mpp=0.25,ref_mpp=0.25,threshold=90, min_size=15, max_size=40,kernel_size=7):
    '''Extract nucleus contours from a patch.
    im : a plt.imread image
    W : -np.log(np.array([hemato_1, eosin, safran, hemato_2]).T / 255) an array containing the staining reference
    model, Lambda, poids : pga model and its parameters
    mpp : patch resolution
    ref_mpp : ref center (usually PB) patch resolution
    verbose : if True save intermediate images
    verbose_path : path to save these images
    g_kernel_size : level of blurring for denoising'''
    # scale hyper-parameters
    g_ker_size_scaled = (g_kernel_size*ref_mpp/mpp)
    threshold_scaled = int(threshold*ref_mpp/mpp)
    min_size_scaled = int(min_size*ref_mpp/mpp)
    max_size_scaled = int(max_size*ref_mpp/mpp)
    scaled_kernel_size = 2 * int((kernel_size*ref_mpp/mpp)/2) + 1 # must be odd and the closest to kernel_size*ref_mpp/mpp
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(scaled_kernel_size,scaled_kernel_size))
    # extract hemalun staining
    V  = vectorize(im)
    im_He = getHstain(V, W, np.maximum((pinv(W) @ V), 0), Lambda, model, poids,n=im.shape[0]) 
    mask = getNucleusMask(im_He)
    clean_mask = getCleanMask(mask, kernel)
    # extract edges
    edges = getEdges(clean_mask,g_ker_size_scaled)
    contour_im0, contours = detectEllipsisContours(im, edges,threshold=threshold_scaled, min_size=min_size_scaled, max_size=max_size_scaled)
    final_im = segmentNucleus(im, contours)
    if verbose: 
        # save intermediate images for debugging
        os.makedirs(verbose_path,exist_ok=True)
        plt.imsave(f"{verbose_path}/im.png", im)
        plt.imsave(f"{verbose_path}/im_He.png", im_He)
        plt.imsave(f"{verbose_path}/edges.png", edges)
        plt.imsave(f"{verbose_path}/contour_im0.png", contour_im0)
        plt.imsave(f"{verbose_path}/final_im.png", final_im)
    return final_im, contours
