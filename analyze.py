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
from scg_detection_tools.dataset import read_dataset_annotation


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
        sub.add_argument("--segment-annotations",
                         dest="seg_annotations",
                         default=None,
                         type=str,
                         help="Use segmentation annotations from a directory. The annotations will be chosen by matching names (without extension) with the image")


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
    hist_parser.add_argument("--detection-boxes",
                             dest="detection_boxes",
                             action="store_true",
                             help="Use boxes from YOLO detection as masks to calculate color histogram")

    return parser.parse_args()


def parse_seg_annotations(imgs, seg_annotations):
    ann_files = None
    img_ann_idx = {}
    # get full path for file
    ann_files = [os.path.join(seg_annotations, f) for f in get_all_files_from_paths(seg_annotations)]
    for i in range(len(ann_files)):
        # Search for a match between annotation file and image file
        # by comparing name without extension
        for j in range(len(imgs)):
            if Path(ann_files[i]).stem == Path(imgs[j]).stem:
                img_ann_idx[imgs[j]] = i
                break # go to next annotation file
    return ann_files, img_ann_idx


def analyze_count(model, detector, imgs, save_detections=False):
    from analysis.object_count import count_per_image

    detections = detector.detect_objects(imgs)
    count = count_per_image(imgs, detections, save=True)

    if save_detections:
        for img, detection in zip(imgs,detections):
            imtools.save_image_detection(default_imgpath=img, detections=detection, save_name=f"count_det{os.path.basename(img)}", save_dir="exp_analysis")

    save_to_csv("count_data.csv", img_id=np.arange(1,len(imgs)+1), count=count)


def analyze_pixel_density(model, detector, imgs, cfg, on_slice=False, seg_annotations=None, save_detections=False):
    from analysis.pixel_density import pixel_density, slice_pixel_density, pixel_density_masks
    from scg_detection_tools.segment import SAM2Segment
    
    seg = SAM2Segment(sam2_ckpt_path=cfg["segment_sam2_ckpt_path"],
                      sam2_cfg=cfg["segment_sam2_cfg"],
                      detection_assist_model=model)
    if on_slice:
        results, densities = slice_pixel_density(img_files=imgs, slice_wh=(640,640), seg=seg)
        detections = [result["detections"] for result in results]

    elif seg_annotations:
        from scg_detection_tools.utils.cvt import contours_to_mask
        ann_files, img_ann_idx = parse_seg_annotations(imgs, seg_annotations)
        ann_contours = []
        ann_masks = []
        for img in imgs:
            ann_file = ann_files[img_ann_idx[img]]
            _, contours = read_dataset_annotation(ann_file)
            ann_contours.append(contours)
            
            imgsz = cv2.imread(img).shape[:2]
            contours_mask = contours_to_mask(contours, imgsz=imgsz)
            ann_masks.append(contours_mask)

        densities = pixel_density_masks(imgs=imgs, imgs_masks=ann_masks)

    else:
        detections = detector.detect_objects(imgs)
        densities = pixel_density(imgs=imgs, detections=detections, save_img_masks=True, seg=seg)

    if save_detections:
        for img, detection in zip(imgs,detections):
            imtools.save_image_detection(default_imgpath=img, detections=detection, save_name=f"pd_det{os.path.basename(img)}", save_dir="exp_analysis")

    img_ids = np.arange(1, len(imgs)+1)
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(img_ids, densities)
    plt.show()


def analyze_color_histogram(model, detector, imgs, raw=False, on_crops=False, on_detection_boxes=False, seg_annotations=None, cspaces=["RGB","HSV"], show=True):
    from analysis.color_hist import rgb_hist, hsv_hist

    ann_files = None
    if seg_annotations:
        ann_files, img_ann_idx = parse_seg_annotations(imgs, seg_annotations)
    
    res_hists = {}
    for img in imgs:
        if raw:
            if "RGB" in cspaces:
                res_hists[img] = rgb_hist(img, show=show)
            if "HSV" in cspaces:
                res_hists[img] = hsv_hist(img, show=show)
        
        detections = None
        if on_crops:
            detections = detector.detect_objects(img)[0]
            res_hists[img] = {"crops": []}
            for box in detections.xyxy.astype(np.int32):
                crop = imtools.crop_box_image(img=img, box_xyxy=box)
                if "RGB" in cspaces:
                    res_hists[img]["crops"].append(rgb_hist(crop, show=show))
                if "HSV" in cspaces:
                    res_hists[img]["crops"].append(hsv_hist(crop, show=show))
        
        if on_detection_boxes:
            if detections is None:
                detections = detector.detect_objects(img)[0]
            if "RGB" in cspaces:
                res_hists[img] = rgb_hist(img=img, boxes=detections.xyxy.astype(np.int32), show=show)
            if "HSV" in cspaces:
                res_hists[img] = hsv_hist(img=img, boxes=detections.xyxy.astype(np.int32), show=show)

        if ann_files is not None:
            ann_file = ann_files[img_ann_idx[img]]
            nclass, contours = read_dataset_annotation(ann_file)

            if "RGB" in cspaces:
                res_hists[img] = rgb_hist(img=img, mask_contours=contours, show=show)
            if "HSV" in cspaces:
                res_hists[img] = hsv_hist(img=img, mask_contours=contours, show=show)

    return res_hists


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
        analyze_count(model, det, img_files, save_detections=args.save_detections)
    elif args.command == "pixel_density":
        analyze_pixel_density(model, det, img_files, on_slice=args.pd_slice, seg_annotations=args.seg_annotations, save_detections=args.save_detections)
    elif args.command == "color_hist":
        analyze_color_histogram(model, det, img_files, raw=args.raw, on_crops=args.ch_on_crop, on_detection_boxes=args.detection_boxes, seg_annotations=args.seg_annotations)
    


if __name__ == "__main__":
    main()

