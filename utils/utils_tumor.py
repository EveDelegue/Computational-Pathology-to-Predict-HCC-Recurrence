import torch
from tqdm import tqdm
import numpy as np
import cv2
import openslide as op
import os
import matplotlib.pyplot as plt
from PIL import Image
from utils.model_archi import IndepResNetModel
from utils.ImageSet import MultiscaleSet
from utils.utils import get_Bright_Dark_perc
from torch.utils.data import DataLoader
import pandas as pds
from torchvision.utils import save_image
from utils.Stain_Normalization import stainNorm_Reinhard

pej_color = np.array([220, 20, 60])  # crimson     #DC143C
non_pej_color = np.array([255, 215, 0])  # gold       #FFD700
healthy_color = np.array([50, 205, 50])  # lime green #32CD32
color2class = {
    tuple(healthy_color): "healthy",
    tuple(non_pej_color): "non pej",
    tuple(pej_color): "pej",
}


def gen_image_from_coords(coords_x, coords_y, y_har, step, colors):
    """
    Generate an image from coordinate points with associated colors.
    This function creates a white image and fills rectangular regions defined by
    coordinate points with colors based on provided labels/predictions.
    Args:
        coords_x (array-like): X-coordinates of the points.
        coords_y (array-like): Y-coordinates of the points.
        y_har (array-like): Labels or predictions for each coordinate point.
        step (int): Size of the square region to fill around each coordinate (step x step).
        colors (dict or array-like): Mapping of labels to RGB color values.
    Returns:
        np.ndarray: Image array of shape (height, width, 3) with dtype uint8 containing
                   the generated image with colored regions. Background is white (255).
    """
    image = 255 + np.zeros(
        (int(coords_y.max()) + 2 * step, int(coords_x.max()) + 2 * step, 3),
        dtype=np.uint8,
    )
    for x, y, p in zip(coords_x, coords_y, y_har):
        image[int(y) : int(y) + step + 1, int(x) : int(x) + step + 1] = colors[p.item()]
    return image


def gen_image_from_coords_bis(coords_x, coords_y, inflams, step):
    """
    Generate an image from coordinate and inflammation data.

    This function creates a white background image and fills it with inflammation
    values at specified coordinates, with each point occupying a step x step region.

    Parameters
    ----------
    coords_x : array-like
        X-coordinates of the points to be placed in the image.
    coords_y : array-like
        Y-coordinates of the points to be placed in the image.
    inflams : array-like
        Inflammation values (RGB tuples or single values) corresponding to each coordinate.
        Must have the same length as coords_x and coords_y.
    step : int
        The size of the square region (step x step) for each point in the image.

    Returns
    -------
    np.ndarray
        A uint8 numpy array of shape (height, width, 3) representing an RGB image.
        The image has a white background (255) with inflammation data painted at the
        specified coordinates. Height and width are calculated based on the maximum
        coordinates plus padding of 2*step.

    Notes
    -----
    - The image is initialized with white (255) values on all channels.
    - Coordinates are converted to integers before indexing.
    - Points near the edges may be clipped based on the image dimensions.
    """
    image = np.zeros(
        (int(coords_y.max()) + 2 * step, int(coords_x.max()) + 2 * step)
    )
    for x, y, p in zip(coords_x, coords_y, inflams):
        image[int(y) : int(y) + step + 1, int(x) : int(x) + step + 1] = p
    return image


def load_models(pth):
    """
    Load multiple pre-trained ResNet34 models.
    This function loads 5 independently trained TripleResNet34 models
    Args:
        pth (str): Directory path containing the model checkpoint files.
    Returns:
        list: A list of 5 IndepResNetModel instances, each loaded with pre-trained weights
            and moved to CUDA device for GPU computation.
    """
    models = []
    for i in tqdm(range(1, 6), desc="loading TripleResnet34"):
        model = IndepResNetModel().cuda()
        weights = torch.load(
            f"{pth}/TripleIndepResNet34_Fold{i}.pt", weights_only=False
        )
        model.load_state_dict(weights["model"])
        models.append(model)
    return models


def get_pred_proba_multi_2(model, data_loader):
    y_proba = torch.zeros(0, dtype=torch.long, device="cpu")
    model.eval()
    with torch.no_grad():
        for inputs in tqdm(data_loader, desc="Prediction NT/NP/P"):
            im1, im2, im3 = inputs
            im1, im2, im3 = (
                im1.cuda(),
                im2.cuda(),
                im3.cuda()
            )
            proba = torch.softmax(model(im1, im2, im3), dim=1)
            y_proba = torch.cat([y_proba, proba.cpu()])
            del proba, inputs, im1, im2, im3
    return y_proba

def get_pred_proba_multi(model, data_loader):
    y_preds = torch.zeros(0, dtype=torch.long, device="cpu")
    y_true = torch.zeros(0, dtype=torch.long, device="cpu")
    y_proba = torch.zeros(0, dtype=torch.long, device="cpu")
    model.eval()
    with torch.no_grad():
        for inputs, target in tqdm(data_loader, desc="Prediction NT/NP/P"):
            im1, im2, im3 = inputs
            im1, im2, im3, target = (
                im1.cuda(),
                im2.cuda(),
                im3.cuda(),
                target.long().squeeze().cuda(),
            )
            proba = torch.softmax(model(im1, im2, im3), dim=1)
            pred = proba.argmax(1).detach()
            y_preds = torch.cat([y_preds, pred.view(-1).cpu()])
            y_true = torch.cat([y_true, target.view(-1).cpu()])
            y_proba = torch.cat([y_proba, proba.cpu()])
            del proba, inputs, target, im1, im2, im3, pred
    return y_true, y_preds, y_proba


def color_transfer(target, reference):
    """
    Transfers color from a reference patch (Paul-Brousse) to a target patch (Mondor or Beaujon) using mean and standard deviation matching in L*a*b* color space.

    Args:
        reference (numpy.ndarray): reference image  (H, W, 3) in RGB.
        target (numpy.ndarray): target image (to transform) (H, W, 3) in RGB.

    Returns:
        numpy.ndarray: Color transferred patch (H, W, 3) in RGB.
    """
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)

    mean_src, std_src = cv2.meanStdDev(reference_lab)
    mean_tgt, std_tgt = cv2.meanStdDev(target_lab)
    mean_src = mean_src.flatten()
    std_src = std_src.flatten()
    mean_tgt = mean_tgt.flatten()
    std_tgt = std_tgt.flatten()

    norm_lab = (target_lab - mean_tgt) * (std_src / (std_tgt + 1e-10)) + mean_src
    norm_lab = np.clip(norm_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(norm_lab, cv2.COLOR_LAB2RGB)


def gen_multiscale_patches(
    slide_name, patches_dir, apply_clr_transfer=False, reference=None
):
    patches = os.listdir(f"{patches_dir}/{slide_name}")
    X = []
    for patch_name in tqdm(patches, desc="read & gen multiscale patches"):
        patch_og = plt.imread(f"{patches_dir}/{slide_name}/{patch_name}")
        if apply_clr_transfer:
            patch = color_transfer(target=patch_og, reference=reference)
            os.makedirs(f"{patches_dir}/{slide_name}",exist_ok=True)
            plt.imsave(f"{patches_dir}/{slide_name}/{patch_name}",patch)
        else:
            patch = patch_og
        res3 = patch.shape[0]  # 1152 626 1094
        res2 = int(res3 / 1.5)
        res1 = int(res2 / 1.5)  # 512 278 486
        x, y = patch.shape[0] // 2, patch.shape[1] // 2
        img_1 = patch[x - res1 // 2 : x + res1 // 2, y - res1 // 2 : y + res1 // 2]
        img_2 = patch[x - res2 // 2 : x + res2 // 2, y - res2 // 2 : y + res2 // 2]
        img_2 = (Image.fromarray(img_2)).resize((res1, res1), Image.Resampling.LANCZOS)
        img_3 = (Image.fromarray(patch)).resize((res1, res1), Image.Resampling.LANCZOS)
        X.append([Image.fromarray(img_1), img_2, img_3])
    return X, list(np.zeros(len(X)))


def compute_mean_predictions(probas_list, mean_type="arithmetic"):
    """Compute predictions using different mean types."""
    all_probas = torch.stack(probas_list, dim=0)

    if mean_type == "arithmetic":
        mean_proba = all_probas.mean(dim=0)
    elif mean_type == "geometric":
        # Use log-sum-exp for numerical stability
        mean_proba = torch.exp(torch.log(all_probas).mean(dim=0))
    elif mean_type == "harmonic":
        eps = 1e-8
        mean_proba = len(probas_list) / ((1 / (all_probas + eps)).sum(dim=0))
    else:
        raise ValueError(f"Unknown mean type: {mean_type}")

    preds = torch.softmax(mean_proba, dim=1).argmax(1)
    return mean_proba, preds


def get_RdYlGr_masks(image):
    # Define the color range to search for
    lower_red, upper_red = pej_color - 10, pej_color + 10
    lower_yellow, upper_yellow = non_pej_color - 10, non_pej_color + 10
    lower_green, upper_green = healthy_color - 10, healthy_color + 10

    # Threshold the image using the color range
    mask_rd = cv2.inRange(image, lower_red, upper_red)
    mask_yl = cv2.inRange(image, lower_yellow, upper_yellow)
    mask_gr = cv2.inRange(image, lower_green, upper_green)

    masked_image_rd = cv2.bitwise_and(image, image, mask=mask_rd)
    masked_image_yl = cv2.bitwise_and(image, image, mask=mask_yl)
    masked_image_gr = cv2.bitwise_and(image, image, mask=mask_gr)
    return masked_image_rd, masked_image_yl, masked_image_gr


def get_largest_connected_area(masked_image, color):
    # Convert the image to grayscale
    gray = cv2.cvtColor(masked_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    # Threshold the image to create a binary mask
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    # Find the contours in the binary mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    if len(contours) != 0:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        # Draw the largest contour on a new image
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        # Apply the mask on the original image
        result = cv2.bitwise_and(masked_image, masked_image, mask=mask)
        # num_pixels = cv2.countNonZero(cv2.inRange(result, color, color))
        area = cv2.contourArea(largest_contour)
    else:
        area = 0.0
        result = mask
    return result, area


def detect_architectures(slide:op.OpenSlide,filtered_coords:list[tuple[int,int]],patch_size_p:tuple[int,int],model_path:str|os.PathLike,mask:np.ndarray,perc_bpx:float=0.3,perc_wpx:float=0.7, verbose:bool=True,verbose_path:str="brouillons/visuals",color_norm:object=stainNorm_Reinhard.DummyNormalizer())->dict[tuple[int,int], dict[str,int]]:
    """Classifies the patchs between 3 classes : non-tumoral, tumoral non-pejorative and tumoral pejorative.
    
    :param slide: input tile  
    :type tile_path: OpenSlide
    :param verbose_path: path for intermediate figures. Default = "brouillons/visuals"
    :type verbose_path: str
    :param verbose: if we wish to show intermediate plots. Default = False 
    :type verbose: bool
    :param perc_bpx: max proportion of black pixels accepted in a patch (default = 0.3)
    :type perc_bpx: float
    param perc_wpx: max proportion of white pixels accepted in a patch (default = 0.7)
    :type perc_wpx: float
    :param filtered_coords: coordinates of the patchs 
    :type filtered_coords: list[tuple]
    :param patch_size_p: patch size 
    :type patch_size_p: tuple[int,int]
    :param model_path: path to the 5 networks used 
    :type model_path: str or PathLike
    :param stain_norm: color normalisation model 
    :type stain_norm: object
    """
    
    # load models
    models = load_models(pth=model_path) # were they trained on an external dataset ?
    # load data
    device="cuda" if torch.cuda.is_available() else "cpu"

    Data = MultiscaleSet(slide,filtered_coords,patch_size_p,device=device,color_norm=color_norm)
    #Data = MultiscaleSet(slide,filtered_coords,patch_size_p,device="cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(Data)
    # create a pandas dataframe for the output
    tumor_dict = {}
    if verbose:
        # create visuals 
        class_dict = {0:0,1:0,2:0}
    # apply model to the data
    with torch.no_grad():
        for img_1,img_2,img_3,x,y in tqdm(loader):
            # check if not too much black or white
            wpx, bpx = get_Bright_Dark_perc(img_3) # compute the black/white ratio
            if wpx < perc_wpx and bpx < perc_bpx: # if not too much white and not too much black
                # compute probabilities for each model
                list_probas = []
                for model in models:
                    model.eval()
                    list_probas.append(torch.softmax(model(img_1, img_2, img_3), dim=1))
                probas = torch.cat(list_probas)
                # ensemble the output with a harmonic mean (see Deep Learning Classification and Quantification of Pejorative and Nonpejorative Architectures in Resected Hepatocellular Carcinoma from Digital Histopathologic Images)
                mean_proba = len(probas) / ((1 / (probas + 1e-8)).sum(dim=0))
                # find more likely class
                preds = int(mean_proba.softmax(0).argmax(0)) 
                # store in a dictionnary
                x,y = int(x),int(y)
                tumor_dict[(x,y)] = {"architecture":preds}
                # visualise result
                if verbose:
                    # show 5 patchs for each class
                    p = np.random.uniform()
                    if (p<(100/len(loader))) and (class_dict[preds] < 5):
                        os.makedirs(os.path.join(verbose_path,f'class_{preds}'),exist_ok=True)
                        save_image(img_3,os.path.join(verbose_path,f'class_{preds}',f'x_{x}_y_{y}.png'))
                        class_dict[int(preds)]+=1
    # visualise whole slide
    if verbose:
        thumbnail = np.array(slide.get_thumbnail(slide.level_dimensions[-1]))
        rescaling_factor = slide.level_downsamples[-1]
        rescaled_p_sz =  (np.array(patch_size_p)//(rescaling_factor)).astype(int)
        for coords,prediction in tumor_dict.items():
            rescaled_coords = (np.array(coords)//rescaling_factor).astype(int)
            thumbnail[rescaled_coords[0]: rescaled_coords[0]+rescaled_p_sz[0], rescaled_coords[1]: rescaled_coords[1]+rescaled_p_sz[1]] = [0,0,0]
            thumbnail[rescaled_coords[0]: rescaled_coords[0]+rescaled_p_sz[0], rescaled_coords[1]: rescaled_coords[1]+rescaled_p_sz[1],prediction["architecture"]] = 255
        plt.imsave(os.path.join(verbose_path,'tumor_pred.png'),thumbnail)

    return tumor_dict


def mask_tumor(result_dict:dict[tuple[int,int], dict[str,int]],patch_size_p:tuple[int,int],slide:op.OpenSlide,verbose:bool=False,verbose_path:str="brouillons/visuals"):
    """Creates a mask of the inside of the tumoral region, as well as the frontier.
    
    :param result_dict: the result of the tumor architecture detection
    :type result_dict: Dict[tuple[int,int], Dict[str,int]]
    :param slide: input tile  
    :type tile_path: OpenSlide
    :param verbose_path: path for intermediate figures. Default = "brouillons/visuals"
    :type verbose_path: str
    :param verbose: if we wish to show intermediate plots. Default = False 
    :type verbose: bool
    :param patch_size_p: patch size 
    :type patch_size_p: tuple[int,int]
    """

    # load thumbnail
    thumbnail = np.array(slide.get_thumbnail(slide.level_dimensions[-1]))
    tumor_thumbnail = np.zeros_like(thumbnail) # 3 chanel image, each chanel will be a class 
    # color it depending on the architectures
    rescaling_factor = slide.level_downsamples[-1]
    rescaled_p_sz =  (np.array(patch_size_p)/(rescaling_factor)).astype(int)
    # init class counter
    class_dict = {0:0,1:0,2:0}
    for coords,prediction in result_dict.items():
            rescaled_coords = (np.array(coords)/rescaling_factor).astype(int)
            # count the frequency of each class 
            class_dict[prediction['architecture']]+=1  
            # put a 1 in the corresponding class
            tumor_thumbnail[rescaled_coords[0]: rescaled_coords[0]+rescaled_p_sz[0], rescaled_coords[1]: rescaled_coords[1]+rescaled_p_sz[1],prediction['architecture']] = 1
    if verbose:
        plt.imsave(os.path.join(verbose_path,'tumor_classes.png'),tumor_thumbnail*255)
    # compute pejorative proportion
    P_ratio = class_dict[2]/(class_dict[2]+class_dict[1]) if (class_dict[2]+class_dict[1])>0 else 0

    ## clean the contours
    # fill in the holes 
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(2*rescaled_p_sz))  
    # clean tissue mask
    closed_tumor = cv2.morphologyEx(tumor_thumbnail, cv2.MORPH_CLOSE, close_kernel)
    if verbose:
        plt.imsave(os.path.join(verbose_path,'tumor_closed.png'),closed_tumor*255)
    
    # remove small objects
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(2*rescaled_p_sz))
    opened_tumor = cv2.morphologyEx(closed_tumor, cv2.MORPH_OPEN, open_kernel)
    if verbose:
        plt.imsave(os.path.join(verbose_path,'tumor_opened.png'),opened_tumor*255)
    
    pej_chanel = opened_tumor[:,:,2]
    non_pej_chanel = opened_tumor[:,:,1]
    non_tum_chanel = opened_tumor[:,:,0]
    # detect largest pej contour
    contours, _ = cv2.findContours(pej_chanel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(pej_chanel)
    if len(contours) != 0:
        # Get the largest area
        largest_contour = max(contours, key=cv2.contourArea)
        # Draw the largest contour on a new image
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        if verbose:
            plt.imsave(os.path.join(verbose_path,'largest_pej.png'),mask)
        area_pej = cv2.contourArea(largest_contour)
    
    # detect largest non-pej contour
    contours, _ = cv2.findContours(non_pej_chanel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(non_pej_chanel)
    if len(contours) != 0:
        # Get the largest area
        largest_contour = max(contours, key=cv2.contourArea)
        # Draw the largest contour on a new image
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        if verbose:
            plt.imsave(os.path.join(verbose_path,'largest_non_pej.png'),mask)
        area_non_pej = cv2.contourArea(largest_contour)
    
    #return in_mask,out_mask
    return 0,0
    