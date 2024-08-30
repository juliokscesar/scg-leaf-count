from typing import Union, List
import numpy as np
import cv2
import matplotlib.pyplot as plt
import warnings

from scg_detection_tools.utils.cvt import contours_to_mask, boxes_to_mask 
from scg_detection_tools.utils.image_tools import mask_img_alpha

def calc_img_hist(img: np.ndarray, channels: List[int], mask: np.ndarray = None):
    hists = []
    for c in channels:
        h = cv2.calcHist([img], [c], mask, [256], [0,256])
        hists.append(h.reshape(256))
    return hists


def rgb_hist(img: Union[str, np.ndarray], 
             mask: np.ndarray = None,
             mask_contours: np.ndarray = None,
             boxes: np.ndarray = None,
             show=True,
             save=False):
    if isinstance(img, str):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return color_hist(img, 
               mask=mask, 
               mask_contours=mask_contours, 
               boxes=boxes, 
               channel_labels=["R","G","B"], 
               channel_colors=["red","green","blue"],
               show=show,
               save=save,
               save_name="rgb_hist.png")


def hsv_hist(img: Union[str, np.ndarray],
             cvt_to_hsv = False,
             mask: np.ndarray = None,
             mask_contours: np.ndarray = None,
             boxes: np.ndarray = None,
             show=True,
             save=False):
    if isinstance(img, str):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if cvt_to_hsv:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    return color_hist(img, 
               cspace="HSV", 
               mask=mask, 
               mask_contours=mask_contours, 
               boxes=boxes, 
               channel_labels=["H","S","V"], 
               show=show,
               save=save, 
               save_name="hsv_hist.png")


def color_hist(img: Union[np.ndarray, str],
               cspace: str = "RGB",
               mask: np.ndarray = None,
               mask_contours: np.ndarray = None,
               boxes: np.ndarray = None,
               channel_labels: List[str] = ["R","G","B"],
               channel_colors: List[str] = ["red", "green", "blue"],
               show=True,
               save=False,
               save_name="color_hist.png"):
    if isinstance(img, str):
        img = cv2.imread(img)
        if cspace == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif cspace == "HSV":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        else:
            raise ValueError(f"Color space {cspace} not supported for histogram")


    if mask and mask_contours:
        raise ValueError("Passed mask and mask_countours, but should use only one of them")

    if mask_contours and boxes:
        warnings.warn("func color_hist: mask_contours and box provided. Using only mask_contours")
    if mask_contours is not None:
        mask = contours_to_mask(contours=mask_contours, imgsz=img.shape[:2])
    elif boxes is not None:
        mask = boxes_to_mask(boxes=boxes, imgsz=img.shape[:2])

    ch_hist = calc_img_hist(img, [0,1,2], mask)
    
    if show:
        fig, axs = plt.subplots(layout="constrained", nrows=1, ncols=2, figsize=(12,8))
        axs[0].imshow(img)

        if mask is not None:
            color = [30, 6, 255]
            alpha = 0.6
            axs[0].imshow(mask_img_alpha(mask, color, alpha))

        axs[0].axis("off")

        for i in range(len(ch_hist)):
            axs[1].plot(ch_hist[i], color=channel_colors[i], label=channel_labels[i])

        axs[1].legend()
        axs[1].set(xlabel="Intensity",
                   ylabel="Frequency")

        plt.show()
        if save:
            fig.savefig(f"exp_analysis/plots/{save_name}")

    return ch_hist

