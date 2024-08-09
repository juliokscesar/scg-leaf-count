import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import yaml
import os
import argparse
from pathlib import Path

import detect
from modelloader import ModelLoader


DETECT_CONFIG={
    "model_type": "yolo",
    
    "yolo_path": "./pretrained_models/yolov8s.pt",

    "rbf_api_key": "",
    "rbf_project": "plant-leaf-detection-att1p",
    "rbf_version": 2,

    "confidence": 50.0,
    "overlap": 50.0,
    "slice_detect": False,
    "slice_wh": (640, 640),
    "slice_overlap_ratio": (0.5, 0.5)
}


def read_config(config_file: str = "analyze_config.yaml"):
    config = {}
    with open(config_file, "r") as f:
        config = yaml.safe_load(config_file)

    # TODO: assert required parameters in config

    return config


def count_data_imgs(img_paths: list[str]) -> np.ndarray:
    config = read_config()
    
    model = ModelLoader(model_type=config["model_type"])
    match DETECT_CONFIG["model_type"]:
        case "yolo":
            model = model.load(path=config["yolov8_custom_model"])

        case "roboflow":
            try:
                rbf_api_key = os.environ["ROBOFLOW_API_KEY"]
            except:
                raise RuntimeError("Setting 'ROBOFLOW_API_KEY' environment variable is required.")

            # Using YOLO-NAS trained version from roboflow
            model = model.load(
                        api_key=rbf_api_key, 
                        project=config["roboflow_project_nas"], 
                        version=config["roboflow_version_nas"]
                    )

        case _:
            raise RuntimeError(f"Model type {config["model_type"]} not supported")

    detect_parameters = config["detect_parameters"]
    count = []
    for img in img_paths:
        num = detect.count_in_image(
                img_path=img,
                model=model,
                confidence=detect_parameters["confidence"],
                overlap=detect_parameters["overlap"],
                slice_detect=detect_parameters["slice_detect"],
                slice_wh=(detect_parameters["slice_w"], detect_parameters["slice_h"]),
                slice_overlap_ratio=(detect_parameters["slice_overlap_ratio_w"], detect_parameters["slice_overlap_ratio_h"])
        )

        count.append(num)
        
        print(f"Counted {num} in image {img}")

    return np.array(count)


"""
Save passed data to CSV

Format of arguments must be:
    name=listdata (e.g. dogs=['pitbull', 'puddle'])
"""
def save_to_csv(out_file: str = "analyze_data.csv", **kwargs):
    df = pd.DataFrame(kwargs)
    df.to_csv(out_file)
    print(f"Saved CSV data to {out_file}")



# TODO: generate analysis given images paths and order.
# support different regression methods (linear, spline, etc)


def leaf_analyze(imgs: list[str], update_cached=False):
    # Read from cached file by default
    # avoid having to count objects in image every time
    if update_cached:
        img_count = count_data_imgs(imgs)
        size = len(img_count)
        days = np.arange(1, size+1)
        save_to_csv(days=days, img_count=img_count)
    else:
        df = pd.read_csv("analyze_data.csv")
        img_count = df["img_count"].to_numpy()
        days = df["days"].to_numpy()
        size = len(img_count)

    print("Assuming first image is Day 1")

    print(f"Counted (Day, Count):\n{[(days[i], img_count[i]) for i in range(size)]}")


    fig, ax = plt.subplot()

    # plot only points (data x img_count)
    ax.scatter(days, img_count, c='b')
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("images_src", 
                        type=str, 
                        nargs='+', 
                        help="Source of images. Can be a single or a list of files, or a directory.")

    parser.add_argument("-c", "--cached", 
                        action="store_true", 
                        help="Use cached data in CSV file. If not specified, avoids having to run detection again and just uses data from before.")
    
    return parser.parse_args()


def main():
    args = parse_args()
    img_src = args.images_src
    cached = args.cached

    img_files = []
    for src in img_src:
        if os.path.isfile(src):
            img_files.append(src)

        elif os.path.isdir(src):
            for (root, _, filenames) in os.walk(src):
                img_files.extend([os.path.join(root, file) for file in filenames])

        else:
            raise RuntimeError(f"{src} is an invalid image source")


    # sort by the name not including extension (because image names are {0..11}.png)
    # if can't convert name to number, use default sort
    def key_sort(item):
        try:
            key = int(Path(item).stem)
        except:
            key = ord(item[0])
        return key

    img_files = sorted(img_files, key=key_sort)

    if cached and not os.path.exists("analyze_data.csv"):
        cached = False

    print(img_files)
    leaf_analyze(img_files, cached)


if __name__ == "__main__":
    main()

