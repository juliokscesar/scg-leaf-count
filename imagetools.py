import supervision as sv
import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_annotated_image(default_imgpath: str, detections: sv.Detections, box_thickness: int=1) -> np.ndarray:
    box_annotator = sv.BoxAnnotator(thickness=box_thickness)
    default_img = cv2.imread(default_imgpath)
    
    annotated_image = box_annotator.annotate(
        scene=default_img.copy(),
        detections=detections
    )

    return annotated_image


def plot_image(img: np.ndarray):
    if img.ndim == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def save_image(img: np.ndarray, name: str, dir: str = "exp"):
    with sv.ImageSink(target_dir_path=dir) as sink:
        sink.save_image(image=img, image_name=name)
    
    print(f"Saved image in ./{dir}/{name}")


def save_image_detection(default_imgpath: str, save_name: str, save_dir: str, detections: sv.Detections, box_thickness: int = 2):
    annotated = generate_annotated_image(default_imgpath, detections, box_thickness)
    save_image(annotated, save_name, save_dir)



def plot_image_detection(default_imgpath: str, detections: sv.Detections, box_thickness: int = 2):
    annotated = generate_annotated_image(default_imgpath, detections, box_thickness)
    plot_image(annotated)

    return annotated
