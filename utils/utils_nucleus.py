import cv2
import torch
import numpy as np
from numpy.linalg import pinv
from skimage import filters


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


def getNucleusMask(im_He, gaussian_filter=(7, 7)):
    blur_c1 = cv2.GaussianBlur(
        cv2.cvtColor(im_He, cv2.COLOR_RGB2GRAY), gaussian_filter, 0
    )
    thresholds = filters.threshold_multiotsu(blur_c1, classes=3)
    multiotsu_mask = np.invert(255 * (blur_c1 > thresholds[1]).astype(np.uint8))
    return multiotsu_mask


def getCleanMask(mask, kernel):
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return opened_mask


def detectContours(im, opened_mask):
    contours, _ = cv2.findContours(
        opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )  # Find contours in the binary image
    contour_im = cv2.drawContours(im.copy(), contours, -1, (0, 0, 0), thickness=2)
    return contour_im, contours


def detectNucleus(contour_im, contours, inf_p=35, inf_a=35):
    perimeters, areas = [], []
    filtred_contours = []
    for contour in contours:
        a, p = cv2.contourArea(contour), cv2.arcLength(contour, True)
        perimeters.append(p)
        areas.append(a)
        if p > inf_p and a > inf_a:
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


def getNucleusFeatures(im, V, W, Lambda, model, kernel, poids):
    N, M = 512, 512
    V = vectorize(im, N=N * M)
    im_He = getHstain(V, W, np.maximum((pinv(W) @ V), 0), Lambda, model, poids)
    mask = getNucleusMask(im_He, gaussian_filter=(7, 7))
    clean_mask = getCleanMask(mask, kernel)
    contour_im0, contours = detectContours(im, clean_mask)
    _, filtred_contours = detectNucleus(contour_im0, contours)
    final_im = segmentNucleus(im, filtred_contours)
    return final_im, filtred_contours
