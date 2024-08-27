import argparse
from typing import List
import supervision as sv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("img_source",
                        nargs="*",
                        help="Source of images. Can be a single image, multiple, or a directory")

    parser.add_argument("-c",
                        "--config",
                        default="analyze_config.yaml",
                        help="Basic configuration options for analysis")
    
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
    print(det_params)
    det = Detector(detection_model=model, detection_params=det_params)

    if not os.path.isdir("exp_analysis"):
        os.mkdir("exp_analysis")

    detections = det.detect_objects(img_files)
    count_per_image(img_files, detections)

    if args.save_detections:
        for detection, img in zip(detections, img_files):
            imtools.save_image_detection(default_imgpath=img, detections=detection, save_name=f"det{os.path.basename(img)}", save_dir="exp_analysis")

if __name__ == "__main__":
    main()

