import argparse
from typing import List, Tuple
import supervision as sv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
from pathlib import Path

from scg_detection_tools.utils.file_handling import(
        read_yaml, get_all_files_from_paths
)
import scg_detection_tools.utils.image_tools as imtools
from scg_detection_tools.models import YOLOv8, YOLO_NAS, RoboflowModel
from scg_detection_tools.detect import Detector
from scg_detection_tools.segment import SAM2Segment


def save_to_csv(out_file: str = "analyze_data.csv", **name_to_data):
    df = pd.DataFrame(name_to_data)
    df.to_csv(out_file)
    print(f"Saved CSV data to {out_file}")


def count_per_image(imgs: List[str], 
                    detections: List[sv.Detections],
                    x_label: str = "image",
                    y_label: str = "count",
                    save = False,
                    show = True):
    count = np.array([len(det.xyxy) for det in detections])
    img_id = np.arange(1, len(imgs)+1)
    save_to_csv(out_file="count_data.csv", img=img_id, count=count)

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(img_id, count, marker='o')

    ax.set(xlabel=x_label,
           ylabel=y_label)

    if save:
        fig.savefig(f"exp_analysis/plots/count_per_image.png")
    if show:
        plt.show()


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
        densities.append(det_pixels / total_pixels)

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

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("img_source",
                        nargs="*",
                        help="Source of images. Can be a single image, multiple, or a directory")

    parser.add_argument("-c",
                        "--config",
                        default="analyze_config.yaml",
                        help="Basic configuration options for analysis")
    
    parser.add_argument("--count",
                        action="store_true",
                        help="Count number of detections and plot it against image ids")
    parser.add_argument("--pixel-density",
                        dest="pixel_density",
                        action="store_true",
                        help="Plot pixel density x image ids by getting number of pixels by segmenting crops of the image using the detections")
    parser.add_argument("--pixel-density-on-slice",
                        dest="pd_slice",
                        action="store_true",
                        help="Plot pixel density but using sliced detection")

    parser.add_argument("--save-detections",
                        dest="save_detections",
                        action="store_true",
                        help="Save image marked with detections")

    return parser.parse_args()

def sort_alphanum(arr):
    def key(item):
        noext = Path(item).stem
        if noext.isnumeric():
            return int(noext)
        else:
            return noext
    return sorted(arr, key=key)

def main():
    args = parse_args()
    
    img_files = get_all_files_from_paths(*args.img_source)
    if len(img_files) == 0:
        raise RuntimeError(f"No files found in {args.img_source}")
    img_files = sort_alphanum(img_files)
    print(img_files)

    cfg = read_yaml(args.config)

    model_type = cfg["model_type"]
    if model_type == "yolov8":
        model = YOLOv8(yolov8_ckpt_path=cfg["yolov8_model_path"])
    elif model_type == "yolonas":
        model = YOLO_NAS(model_arch=cfg["yolonas_arch"],
                         classes=cfg["data_classes"],
                         checkpoint_path=cfg["yolonas_model_path"])
    elif model_type == "roboflow":
        model = RoboflowModel(api_key=os.getenv("ROBOFLOW_API_KEY"),
                              project=cfg["roboflow_project_nas"],
                              version=cfg["roboflow_version_nas"])
    
    det_params = cfg["detect_parameters"]
    det_params["embed_slice_callback"] = None
    det = Detector(detection_model=model, detection_params=det_params)
    seg = SAM2Segment(sam2_ckpt_path=cfg["segment_sam2_ckpt_path"],
                      sam2_cfg=cfg["segment_sam2_cfg"],
                      detection_assist_model=model)

    os.makedirs("exp_analysis/plots", exist_ok=True)

    if args.pd_slice:
        results = slice_pixel_density(img_files, slice_wh=(640,640), seg=seg)
        detections = [result["detections"] for result in results]
    else:
        detections = det.detect_objects(img_files)


    if args.count:
        count_per_image(img_files, detections, save=True)

    if args.pixel_density:
        pixel_density(img_files, detections, on_crops=False, seg=seg, save=True)

    if args.pd_slice:
        slice_pixel_density(img_files, slice_wh=(640,640), seg=seg)
        
    if args.save_detections:
        for detection, img in zip(detections, img_files):
            imtools.save_image_detection(default_imgpath=img, detections=detection, save_name=f"det{os.path.basename(img)}", save_dir="exp_analysis")

if __name__ == "__main__":
    main()

