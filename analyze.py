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
import imagetools
import utils


def read_config(config_file: str = "analyze_config.yaml"):
    config = utils.read_yaml(config_file)
    return config


def count_data_imgs(img_paths: list, save_annotated_img: bool = False) -> np.ndarray:
    config = read_config()
    
    model = ModelLoader(model_type=config["model_type"])

    if config["model_type"] == "yolo":
        model = model.load(path=config["yolov8_custom_model"])
    
    elif config["model_type"] == "yolonas":
        model = model.load(model_arch=config["yolonas_arch"],
                           num_classes=config["data_num_classes"],
                           chkpt_path=config["yolonas_model_path"])

    elif config["model_type"] == "roboflow":
        try:
            rbf_api_key = os.environ["ROBOFLOW_API_KEY"]
        except:
            raise RuntimeError("Setting 'ROBOFLOW_API_KEY' environment variable is required.")
        # Using YOLO-NAS trained version from roboflow
        model = model.load(api_key=rbf_api_key, 
                           project=config["roboflow_project_nas"], 
                           version=config["roboflow_version_nas"])

    else:
        raise RuntimeError(f"Model type {config['model_type']} not supported")

    detect_parameters = config["detect_parameters"]
    count = []
    for img in img_paths:
        # Special parameters key: 'img' + {0..11} (config file has format img0, img1...)
        spec_name = "img" + Path(img).stem
        if config["use_spec_detect_parameters"] and spec_name in config["spec_detect_parameters"]:
            img_parameters = config["spec_detect_parameters"][spec_name]

            detections = detect.detect_objects(img_path=img,
                                               model=model,
                                               confidence=img_parameters["confidence"],
                                               overlap=img_parameters["overlap"],
                                               slice_detect=img_parameters["slice_detect"],
                                               slice_wh=(img_parameters["slice_w"], img_parameters["slice_h"]),
                                               slice_overlap_ratio=(img_parameters["slice_overlap_ratio_w"], img_parameters["slice_overlap_ratio_h"]))

        else:
            detections = detect.detect_objects(img_path=img,
                                               model=model,
                                               confidence=detect_parameters["confidence"],
                                               overlap=detect_parameters["overlap"],
                                               slice_detect=detect_parameters["slice_detect"],
                                               slice_wh=(detect_parameters["slice_w"], detect_parameters["slice_h"]),
                                               slice_overlap_ratio=(detect_parameters["slice_overlap_ratio_w"], detect_parameters["slice_overlap_ratio_h"]))

        num = detect.count_objects(detections)
        count.append(num)

        if (save_annotated_img):
            imagetools.save_image_detection(default_imgpath=img,
                                            save_name="analyzed_" + os.path.basename(img),
                                            save_dir="exp_analysis",
                                            detections=detections)

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



def leaf_analyze(imgs: list = None, show=False, use_cached=False, save_annotated_img=False, cache_file: str = "analyze_data.csv"):
    # Read from cached file by default
    # avoid having to count objects in image every time
    if not use_cached and imgs is not None:
        img_count = count_data_imgs(imgs, save_annotated_img)
        size = len(img_count)
        days = np.arange(1, size+1)
        save_to_csv(days=days, img_count=img_count)
    else:
        df = pd.read_csv(cache_file)
        img_count = df["img_count"].to_numpy()
        days = df["days"].to_numpy()
        size = len(img_count)


    # output is a matrix len(days)x2 where mat[0] = (day0,img_count0)
    return np.array((days, img_count)).T


####################################################
################  PLOT FUNCTIONS  ##################
####################################################

def plot(x: np.ndarray, 
         y: np.ndarray,
         save_name: str,
         **axset_kwargs):
    fig, ax = plt.subplots(layout="constrained")
    
    ax.plot(x, y, marker='o')
    
    ax.set(**axset_kwargs)

    fig.savefig(f"exp_analysis/plots/{save_name}.png")

# datas must be list of lists of points (x,y)
def multi_plot(datas, 
               names: list[str], 
               nrows: int, 
               ncols: int,
               save_name: str,
               **axset_kwargs):
    multi_fig, axs = plt.subplots(nrows=nrows, 
                                  ncols=ncols,
                                  figsize=(15,12),
                                  gridspec_kw={"wspace": 0.05},
                                  layout="constrained")

    count = 0
    for data, ax in zip(datas, axs.flat):
        x = [point[0] for point in data]
        y = [point[1] for point in data]

        ax.plot(x, y, marker='o')
        ax.text(0.5, 0.95, f"({names[count]})", transform=ax.transAxes, fontsize=14, verticalalignment="top")
        count += 1

        ax.set(**axset_kwargs)

    multi_fig.savefig(f"exp_analysis/plots/{save_name}.str")

"""
Plot the average of a list of datas.
Each element of the data list should be an array of tuples of
that data. 
If the data contains N sets of (xi,yi), then data[0]=(x0,y0)...
so datas[0] = data[0] => datas[0] is the first set of tuples of the data.

To plot the average of those points across different sets of data,
every independent variable should be the same across same 'indices'. So datas[0][0][x] == datas[1][0][x], for a set where 'x' is the independent variable.
"""
def average_plot(datas, error_bar = True, save_name: str = "avg_plot", dependent_var_idx = 1, **axset_kwargs):
    x_vals = (datas[0].T)[0]
    
    # create a matrix where each column is an x value
    # and each row is the corresponding y value across all data
    data_mat = np.empty(shape=(len(datas), len(x_vals)))
    for row, data in enumerate(datas):
        x_data = [point[0] for point in data]
        y_data = [point[1] for point in data]
        
        x_idx = np.searchsorted(x_vals, x_data)
        data_mat[row, x_idx] = y_data

    avg_data = np.mean(data_mat, axis=0)

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(x_vals, avg_data, marker='o')

    if error_bar:
        std_err = np.std(data_mat, axis=0)
        ax.errorbar(x_vals, avg_data, yerr=std_err, c='b')

    ax.set(**axset_kwargs)
    fig.savefig(f"exp_analysis/plots/{save_name}.png")


def polar_plot(x: list, y: list):
    theta = np.linspace(0, 2*np.pi, len(x))
    r = y

    fig, ax = plt.subplots()
    ax.plot(theta, r)


def stem_plot(x: list, y: list):
    fig, ax = plt.subplots()
    ax.stem(x, y)

def quiver_plot(x, y):
    u = np.diff(x)
    v = np.diff(y)
    fig, ax = plt.subplots()
    ax.quiver(x[:-1], y[:-1], u, v)

####################################################

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("images_src", 
                        type=str, 
                        nargs='*',
                        default=None, 
                        help="Source of images. Can be a single or a list of files, or a directory.")

    parser.add_argument("-c", "--cached", 
                        action="store_true", 
                        help="Use cached data in CSV file. If specified, avoids having to run detection again and just uses data from before.")

    parser.add_argument("--cache-file",
                        dest="cache_file",
                        default="analyze_data.csv",
                        help="CSV containing data. Default uses 'analyze_data.csv'")

    parser.add_argument("--show", 
                        action="store_true", 
                        dest="show", 
                        help="Plot data and show")
    
    parser.add_argument("--save-detections",
                        action="store_true",
                        dest="save_detections",
                        help="Save image annotated with bounding boxes after detecting.")

    parser.add_argument("--gen-plots",
                        action="store_true",
                        dest="gen_plots",
                        help="Generate plots using analyze data")
    
    return parser.parse_args()


def main():
    args = parse_args()
    img_src = args.images_src
    cached = args.cached
    cache_file = args.cache_file
    show = args.show
    save_detections = args.save_detections
    gen_plots = args.gen_plots

    img_files = None
    if img_src:
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

    if cached:
        assert(os.path.isfile(cache_file))
    

    data = leaf_analyze(img_files, show=show, use_cached=cached, save_annotated_img=save_detections, cache_file=cache_file)
    if gen_plots:
        if not os.path.isdir("exp_analysis/plots"):
            os.makedirs("exp_analysis/plots")

        plot(x=data.T[0],
             y=data.T[1],
             save_name="analyze_plot",
             xlabel="Days",
             ylabel="Leaf count")


if __name__ == "__main__":
    main()

