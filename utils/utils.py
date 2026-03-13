import cv2
import torch
import openslide
from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter


def log_tick_formatter(x, pos):
    return f"{np.expm1(x):.0f}"


def interpolate(size: int, nominateur: float, denominateur: float):
    """
    Calculate the interpolated size based on the ratio between two resolutions.

    This function adjusts a given size by scaling it according to the ratio of two
    microscopy image resolutions. It is commonly used to convert dimensions between
    different scanner formats (e.g., 3D Histopathology vs. Hamamatsu/Leica scanners).

    Args:
        size (int): The original dimension size (typically 1152, 1024, or 512 pixels).
        nominateur (float): The source resolution in micrometers per pixel
                           (e.g., 0.25 for 3D Histopathology - Paul-Brousse).
        denominateur (float): The target resolution in micrometers per pixel
                             (e.g., 0.46 for Hamamatsu or 0.26 for Leica - Mondor/Beaujon).

    Returns:
        int: The scaled size adjusted for the resolution ratio.

    Example:
        >>> interpolate(1152, 0.25, 0.46)
        625
    """
    return int((size * nominateur) // denominateur)


def get_BrightandDark_perc(pil_image, bright_threshold=200, dark_threshold=20):
    """
    Calculate the percentage of bright and dark pixels in an image.

    This function converts a PIL image to grayscale and computes the proportion
    of pixels that are considered "bright" (white) and "dark" (black) based on
    specified intensity thresholds.

    Args:
        pil_image (PIL.Image): A PIL Image object to analyze.
        bright_threshold (int, optional): The grayscale intensity threshold above which
            pixels are considered bright. Defaults to 200.
        dark_threshold (int, optional): The grayscale intensity threshold below which
            pixels are considered dark. Defaults to 20.

    Returns:
        tuple: A tuple containing:
            - bright_percentage (float): The proportion of bright pixels (0.0 to 1.0).
            - dark_percentage (float): The proportion of dark pixels (0.0 to 1.0).

    Example:
        >>> from PIL import Image
        >>> img = Image.open('sample.jpg')
        >>> bright_pct, dark_pct = get_BrightandDark_perc(img)
        >>> print(f"Bright: {bright_pct:.2%}, Dark: {dark_pct:.2%}")
    """
    grayscale_image = np.array(pil_image.convert("L"))
    # Count bright/white and dark/black pixels
    bright_pixels = np.sum(grayscale_image > bright_threshold)
    dark_pixels = np.sum(grayscale_image < dark_threshold)
    # Calculate percentages
    total_pixels = grayscale_image.size
    bright_percentage = bright_pixels / total_pixels
    dark_percentage = dark_pixels / total_pixels
    return bright_percentage, dark_percentage


def detect_tissue_regions(array_slide):
    """
    Detect the largest tissue region in a slide image and return its bounding rectangle.
    Args:
        array_slide (np.ndarray): A slide image as a NumPy array in RGB color format.
    Returns:
        np.ndarray: A 1D array containing [x, y, w, h] where:
            - x (int): The x-coordinate of the top-left corner of the bounding rectangle
            - y (int): The y-coordinate of the top-left corner of the bounding rectangle
            - w (int): The width of the bounding rectangle
            - h (int): The height of the bounding rectangle
    """
    gray = cv2.cvtColor(array_slide, cv2.COLOR_RGB2GRAY)
    _, binary_mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )  # Threshold the image to create a binary mask
    output = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )  # Find contours in the binary mask
    contours = output[0] if len(output) == 2 else output[1]
    largest_contour = max(
        contours, key=cv2.contourArea
    )  # Find the contour with the largest area (largest tissue region)
    x, y, w, h = cv2.boundingRect(
        largest_contour
    )  # Get the coordinates of the bounding rectangle around the largest tissue region
    return np.array([x, y, w, h])


def image_stats(image: np.ndarray):
    """Compute the mean and standard deviation of each channel in the image."""
    mean, std = cv2.meanStdDev(image)
    mean = mean.flatten()
    std = std.flatten()
    return mean, std


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


def generate_patches_from_wsi(
    slide_name,
    path_to_wsi,
    patch_size,
    step,
    path_to_patches,
    vis_scale,
    overview_path,
    coords_path,
    perc_bpx=0.05,
    perc_wpx=0.85,
    enlarge=5,
    verbose=False,
):
    real_enlarge = int(enlarge / vis_scale)
    slide_name = f"{path_to_wsi}/{slide_name}"
    slide = openslide.OpenSlide(f"{slide_name}")
    print(slide.dimensions)
    slide_w, slide_h = slide.dimensions
    W, H = int(slide_w * vis_scale), int(slide_h * vis_scale)
    array_slide = np.array(slide.get_thumbnail((W, H)).convert("RGB"))
    xywh = detect_tissue_regions(array_slide)
    real_x, real_y, real_w, real_h = np.array(xywh // vis_scale, np.int64)
    x, y, width, height = xywh
    x_start, y_start, x_end, y_end = (
        max(0, x - enlarge),
        max(0, y - enlarge),
        x + enlarge,
        y + enlarge,
    )
    real_x, real_y, real_w, real_h = (
        max(0, real_x - real_enlarge),
        max(0, real_y - real_enlarge),
        real_w + real_enlarge,
        real_h + real_enlarge,
    )
    scaled_slide = np.array(array_slide)[
        y_start : y_end + height, x_start : x_end + width
    ]

    slide_name = slide_name.split("/")[-1].split(".")[0]
    coords_x, coords_y = [], []
    range_x = range(real_x, real_x + real_w, step)
    range_y = range(real_y, real_y + real_h, step)
    pth1 = path_to_patches + slide_name.split(".")[0]
    Path(pth1).mkdir(exist_ok=True)
    with tqdm(total=len(range_x) * len(range_y), desc="Patch extraction") as pbar:
        for x in range_x:
            for y in range_y:
                patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert(
                    "RGB"
                )
                wpx, bpx = get_BrightandDark_perc(patch)
                if wpx < perc_wpx and bpx < perc_bpx:
                    coords_x.append(x)
                    coords_y.append(y)
                    patch.save(f"{pth1}/patch_x_{str(x)}_y_{str(y)}.jpg")
                    pbar.update(1)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    fig.suptitle(slide_name, fontsize=18)
    ax1.imshow(array_slide)
    ax1.add_patch(Rectangle(xy=xywh[:2], width=xywh[-2], height=xywh[-1], edgecolor="k", lw=1.5, facecolor="none"))  # type: ignore
    ax2.imshow(scaled_slide)
    ax3.scatter(
        np.array(coords_x) * vis_scale - x_start,
        np.array(coords_y) * vis_scale - y_start,
        c="g",
        marker="s",
        s=8,
        alpha=0.3,
        edgecolor="yellow",
    )
    ax3.imshow(scaled_slide)
    for ax in [ax1, ax2, ax3]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{overview_path}/{slide_name}_overview.png", dpi=150)
    if verbose:
        plt.show()

    to_save = {
        "coords_x": coords_x,
        "coords_y": coords_y,
        "xy_start_end": [x_start, y_start, x_end, y_end],
        "xywh_real": [real_x, real_y, real_w, real_h],
        "scaled_slide": scaled_slide,
        "vis_scale": vis_scale,
    }

    handle = f"{coords_path}/{slide_name}_coords_checkpoint.pt"
    torch.save(to_save, handle)
    print(slide_name, "done!")
