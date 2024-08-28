import supervision as sv
import numpy as np
from scg_detection_tools.segment import SAM2Segment
from typing import List, Tuple

# To count pixel quantity of detections, use segmentation on crops
# and count the quantity of pixels in every crop (= pixels of every leaf)
def pixel_density(imgs: List[str], 
                  detections: List[sv.Detections], 
                  seg: SAM2Segment,
                  on_crops=False,
                  x_label="img_id", 
                  y_label="pixel density",
                  save=False,
                  show=True):
    img_id = np.arange(1, len(imgs)+1)
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
                if (len(masks) != 1):
                    print(f"DEBUG: one crop with len(masks)={len(masks)}")

                det_pixels += mask_pixels(masks[0])

        else:
            masks = seg._segment_detection(img, detection)
            for mask in masks:
                det_pixels += mask_pixels(mask)

        print(f"DEBUG: det_pixels={det_pixels}")
        densities.append((det_pixels / total_pixels) * 100.0)

    save_to_csv(out_file="pixel_density_data.csv", img=img_id, densities=densities)

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(img_id, densities, marker='o')
    ax.set(xlabel=x_label, ylabel=y_label)

    if show:
        plt.show()
    if save:
        fig.savefig("exp_analysis/plots/pixel_density.png")


def slice_pixel_density(img_files: List[str], 
                        slice_wh: Tuple[int,int], 
                        seg: SAM2Segment,
                        x_label="img_id",
                        y_label="pixel density",
                        show=True,
                        save=False):
    img_id = np.arange(1, len(img_files)+1)
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
        
        density = (obj_pixels/total_pixels)*100.0
        densities.append(density)
        print(f"Image: {img}, Total pixels: {total_pixels}, Object pixels: {obj_pixels}, Density: {density}%")

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(img_id, densities, marker='o')
    ax.set(xlabel=x_label,
           ylabel=y_label)
    if show:
        plt.show()
    if save:
        fig.savefig("exp_analysis/plots/slice_pixel_density.png")

    return results


def mask_pixels(mask: np.ndarray):
    binary_mask = np.where(mask > 0.5, 1, 0)
    pixels = int(np.sum(binary_mask == 1))
    return pixels

