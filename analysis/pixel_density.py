import supervision as sv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scg_detection_tools.segment import SAM2Segment
from typing import List, Tuple, Union

from scg_detection_tools.utils.image_tools import mask_img_alpha, segment_annotated_image, save_image

# To count pixel quantity of detections, use segmentation on crops
# and count the quantity of pixels in every crop (= pixels of every leaf)
def pixel_density(imgs: List[str], 
                  detections: List[sv.Detections], 
                  seg: SAM2Segment,
                  on_crops=False,
                  save_img_masks=True,
                  show=True):
    densities = []
    for img_path, detection in zip(imgs, detections):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        total_pixels = img.size // img.shape[2]
        print(f"{img_path} img shape: {img.shape}. Size:{total_pixels}. HxW={img.shape[0] * img.shape[1]}")

        det_pixels = 0
        if on_crops:
            for box in detection.xyxy.astype(np.int32):
                crop = imtools.crop_box_image(img, box)
                ch, cw, _ = crop.shape
                mid_point = [cw//2, ch//2]
                masks = seg._segment_point(img_p=crop,
                                           input_points=np.array([mid_point]),
                                           input_labels=np.array([1]))
                det_pixels += mask_pixels(masks[0])

        else:
            masks = seg._segment_detection(img, detection)
            if save_img_masks:
                #ann_img = segment_annotated_image(img, masks)
                #save_image(ann_img, name=f"mask_{img_path}.png", dir="exp_analysis/masked", cvt_to_bgr=True)
                pass
            for mask in masks:
                det_pixels += mask_pixels(mask)
        densities.append((det_pixels / total_pixels))

    return densities



def slice_pixel_density(img_files: List[str], 
                        slice_wh: Tuple[int,int], 
                        seg: SAM2Segment,
                        x_label="img_id",
                        y_label="pixel density",
                        save_img_masks=False):
    densities = []
    results = []
    for img in img_files:
        slice_seg = seg.slice_segment_detect(img_path=img, slice_wh=slice_wh)
        results.append(slice_seg)
        
        total_pixels = cv2.imread(img).size // 3
        obj_pixels = 0
        for slice in slice_seg["slices"]:
            masks_pixels = 0
            for mask in slice["masks"]:
                masks_pixels += mask_pixels(mask)
            obj_pixels += masks_pixels
        
        density = (obj_pixels/total_pixels)
        densities.append(density)
        print(f"Image: {img}, Total pixels: {total_pixels}, Object pixels: {obj_pixels}, Density: {density}%")

    return results, densities


def pixel_density_masks(imgs: List[str], 
                        imgs_masks: np.ndarray, 
                        x_label="Image ID", 
                        y_label="Pixel density",
                        show=True,
                        save=False):
    densities = []
    for img, masks in zip(imgs, imgs_masks):
        img_pixels = cv2.imread(img).size // 3

        masks_pixels = 0
        for mask in masks:
            masks_pixels += mask_pixels(mask)
        density = masks_pixels / img_pixels
        
        if show:
            fig, ax = plt.subplots()
            ig = cv2.imread(img)
            ax.imshow(cv2.cvtColor(ig, cv2.COLOR_BGR2RGB))
            for mask in masks:
                ax.imshow(mask_img_alpha(mask, color=[30,6,255],alpha=0.5))
            ax.set_title(f"{density}")
            plt.show()

        densities.append(masks_pixels / img_pixels)

    return densities


def mask_pixels(mask: np.ndarray):
    binary_mask = np.where(mask > 0.5, 1, 0)
    pixels = int(np.sum(binary_mask == 1))
    return pixels

