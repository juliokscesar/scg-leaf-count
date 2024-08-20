import supervision as sv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils

def generate_annotated_image(default_imgpath: str, detections: sv.Detections, box_thickness: int=1) -> np.ndarray:
    box_annotator = sv.BoxAnnotator(thickness=box_thickness)
    default_img = cv2.imread(default_imgpath)
    
    annotated_image = box_annotator.annotate(
        scene=default_img.copy(),
        detections=detections
    )

    return annotated_image


def plot_image(img: np.ndarray, convert_to_rgb=True):
    if img.ndim == 2:
        plt.imshow(img, cmap="gray")
    elif convert_to_rgb:
        if img.shape[-1] == 4:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGBA2RGB))
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img)
    plt.axis("off")
    plt.show()


def save_image(img: np.ndarray, name: str, dir: str = "exp", convert_to_BGR=False):
    if convert_to_BGR:
        if img.shape[:-1] == 4: # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    with sv.ImageSink(target_dir_path=dir) as sink:
        sink.save_image(image=img, image_name=name)

    print(f"Saved {dir}/{name}")
    

def load_imgs(*args, color_space="RGB"):
    files = utils.get_all_files_from_paths(*args)
    imgs = []
    for file in files:
        img = cv2.imread(file)

        match color_space:
            case "RGB":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                break
            
            case "RGBA":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
                break

        imgs.append(img)

    return imgs

def save_image_detection(default_imgpath: str, save_name: str, save_dir: str, detections: sv.Detections, box_thickness: int = 2):
    annotated = generate_annotated_image(default_imgpath, detections, box_thickness)
    save_image(annotated, save_name, save_dir)



def plot_image_detection(default_imgpath: str, detections: sv.Detections, box_thickness: int = 2):
    annotated = generate_annotated_image(default_imgpath, detections, box_thickness)
    plot_image(annotated)

    return annotated


def crop_box(img, bottom_right: tuple[int, int], top_left: tuple[int, int]):
    if isinstance(img, str):
        img = cv2.imread(img)

    row0, col0 = top_left
    row1, col1 = bottom_right
    return img[col0:(col1+1), row0:(row1+1)]

