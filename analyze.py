import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import yaml
import os

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




# TODO: generate analysis given images paths and order.
# support different regression methods (linear, spline, etc)


def analyze(imgs: list[str]):
    img_count = count_data_imgs(imgs)
    size = len(img_count)
    days = np.arange(size)

    print("Assuming first image is Day 0")

    print(f"Counted (Day, Count):\n{[(days[i], img_count[i]) for i in range(size)]}")


    linfit= stats.linregress(days, img_count)
    print(f"ax+b: a={linfit.slope}, b={linfit.intercept}")

