import argparse
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


def save_to_csv(out_file: str = "analyze_data.csv", **name_to_data):
    df = pd.DataFrame(name_to_data)
    df.to_csv(out_file)
    print(f"Saved CSV data to {out_file}")


def add_common_args(*subparsers):
    for sub in subparsers:
        sub.add_argument("img_source",
                         nargs="*",
                         help="Source of images. Can be a single image, multiple, or a directory")
        sub.add_argument("-c",
                         "--config",
                         default="analyze_config.yaml",
                         help="Basic configuration options for analysis")
        sub.add_argument("--save-detections",
                         dest="save_detections",
                         action="store_true",
                         help="Save image marked with detections")

def parse_args():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest="command")

    count_parser = subparser.add_parser("count", help="Count objects in images and plot count per images")
    pd_parser = subparser.add_parser("pixel_density", help="Calculate pixel density using segmentation and plot density per images")
    hist_parser = subparser.add_parser("color_hist", help="Plot color histogram of images or crops")
    add_common_args(count_parser, pd_parser, hist_parser)
    
    pd_parser.add_argument("--on-slice",
                           dest="pd_slice",
                           action="store_true",
                           help="Plot pixel density but using sliced detection")
    
    hist_parser.add_argument("--on-crops",
                             dest="ch_on_crop",
                             action="store_true",
                             help="Color histogram on crops using YOLO detection boxes")
    hist_parser.add_argument("--raw",
                             action="store_true",
                             help="Plot color histogram of image as it is")
    # TODO: add some option like '--contour-files' that loads the annotations of every contour in the image
    # taking the contour from the .txt file with the same name as the image (without extension)

    return parser.parse_args()


def analyze_count(args, model, detector, imgs):
    from analysis.object_count import count_per_image

    detections = detector.detect_objects(imgs)
    count_per_image(imgs, detections, save=True)

    if args.save_detections:
        for img, detection in zip(imgs,detections):
            imtools.save_image_detection(default_imgpath=img, detections=detection, save_name=f"count_det{os.path.basename(img)}", save_dir="exp_analysis")


def analyze_pixel_density(args, model, detector, imgs):
    from analysis.pixel_density import pixel_density, slice_pixel_density
    
    seg = SAM2Segment(sam2_ckpt_path=cfg["segment_sam2_ckpt_path"],
                      sam2_cfg=cfg["segment_sam2_cfg"],
                      detection_assist_model=model)
    if args.pd_slice:
        results = slice_pixel_density(img_files=imgs, slice_wh=(640,640), seg=seg)
        detections = [result["detections"] for result in results]
    else:
        detections = detector.detect_objects(imgs)
        pixel_density(imgs=imgs, detections=detections, save=True)

    if args.save_detections:
        for img, detection in zip(imgs,detections):
            imtools.save_image_detection(default_imgpath=img, detections=detection, save_name=f"pd_det{os.path.basename(img)}", save_dir="exp_analysis")

def analyze_color_histogram(args, model, detector, imgs):
    from analysis.color_hist import color_hist

    for img in imgs:
        if args.raw:
            color_hist(img)

        if args.ch_on_crop:
            detections = detector.detect_objects(img)[0]
            for box in detections.xyxy.astype(np.int32):
                crop = imtools.crop_box_image(img=img, box_xyxy=box)
                color_hist(crop)


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

    os.makedirs("exp_analysis/plots", exist_ok=True)

    if args.command == "count":
        analyze_count(args, model, det, img_files)
    elif args.command == "pixel_density":
        analyze_pixel_density(args, model, det, img_files)
    elif args.command == "color_hist":
        analyze_color_histogram(args, model, det, img_files)
    


if __name__ == "__main__":
    main()

