import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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


def count_data_imgs(img_paths: list[str]) -> np.ndarray:
    model = ModelLoader(model_type=DETECT_CONFIG["model_type"])
    match DETECT_CONFIG["model_type"]:
        case "yolo":
            model = model.load(path=DETECT_CONFIG["yolo_path"])

        case "roboflow":
            model = model.load(
                        api_key=DETECT_CONFIG["rbf_api_key"], 
                        project=DETECT_CONFIG["rbf_project"], 
                        version=DETECT_CONFIG["rbf_version"]
                    )

        case _:
            raise Exception(f"Model type {DETECT_CONFIG["model_type"]} not supported")

    count = []
    for img in img_paths:
        num = detect.count_in_image(
                img_path=img,
                model=model,
                confidence=DETECT_CONFIG["confidence"],
                overlap=DETECT_CONFIG["overlap"],
                slice_detect=DETECT_CONFIG["slice_detect"],
                slice_wh=DETECT_CONFIG["slice_wh"],
                slice_overlap_ratio=DETECT_CONFIG["slice_overlap_ratio"]
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

