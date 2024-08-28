from typing import Union, List
import numpy as np
import cv2
import matplotlib.pyplot as plt

from scg_detection_tools.utils.cvt import cvt_contours_to_mask

def calc_rgb_hist(rgb_img: np.ndarray, channels: List[int], mask: np.ndarray = None):
    hists = []
    for c in channels:
        h = cv2.calcHist([rgb_img], [c], mask, [256], [0,256])
        hists.append(h)
    return hists

def color_hist(img: Union[str, np.ndarray], 
               mask: np.ndarray = None,
               mask_contours: np.ndarray = None,
               save=False):
    if isinstance(img, str):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if mask and mask_countors:
        raise ValueError("Passed mask and mask_countours, but should use only one of them")

    if mask_contours:
        mask = cvt_contours_to_mask(contours=mask_contours, imgsz=img.shape[:2])

    r_hist, g_hist, b_hist = calc_rgb_hist(img, [0,1,2], mask)
    
    fig, axs = plt.subplots(layout="constrained", nrows=1, ncols=2, figsize=(12,8))
    axs[0].imshow(img)
    axs[0].axis("off")

    axs[1].plot(r_hist, color="red", label="R")
    axs[1].plot(g_hist, color="green", label="G")
    axs[1].plot(b_hist, color="blue", label="B")

    plt.show()
    if save:
        fig.savefig("exp_analysis/plots/color_hist.png")


