import os
import sys
from pathlib import Path

_GN_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
print(_GN_ROOT_PATH)
sys.path.append(_GN_ROOT_PATH)


from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
import sam2
import sam2.build_sam
import sam2.sam2_image_predictor
import argparse

import detect
import utils
import imagetools


class GenMethod(Enum):
    SAM2_YOLO_SEGMENT = 1
    YOLO_ASSIST = 2
    MANUAL_SEGMENT = 3

class Generator:
    def __init__(self, 
                 method: GenMethod = GenMethod.YOLO_ASSIST,
                 **kwargs):
        self._method = method

        self._sam2_predictor = None
        if self._method == GenMethod.SAM2_YOLO_SEGMENT:
            if "sam2_chkpt_path" not in kwargs: 
                self.load_sam2()
            else:
                self.load_sam2(sam2_chkpt_path=kwargs["sam2_chkpt_path"])

    def __call__(self):
        pass

    # TODO: Manual segment:
    # use different individual and group images of leafs
    # segment them manually
    # load the image and the segmentation data
    # randomize a lot of them in a background image
    # keep track of every segmentation transformation,
    # including translation, rotation
    

    # TODO: SAM2_YOLO_SEGMENT:
    # use YOLO to first detect the bounding box of the object
    # then pass that bounding box to SAM2 to segment the individual leaf
    # save that segmentation in YOLO-dataset format
    # fine tune model with that dataset
    # => can also fine tune sam2 to make better segmentations
    # make slices of 640x640
    def load_sam2(self, sam2_chkpt_path: str = f"{_GN_ROOT_PATH}/dataset_generator/sam2chkpts/sam2_hiera_tiny.pt", sam2_cfg: str = "sam2_hiera_t.yaml"):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        sam2_model = sam2.build_sam.build_sam2(sam2_cfg, sam2_chkpt_path, device=device)
        predictor = sam2.sam2_image_predictor.SAM2ImagePredictor(sam2_model)
        self._sam2_predictor = predictor


    def sam2_segment(self, img_path: str, bounding_boxes: np.ndarray):
        if self._sam2_predictor is None:
            raise RuntimeError("Need to load sam2 model first")
        
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)

        self._sam2_predictor.set_image(img)
        masks, _, _ = self._sam2_predictor.predict(point_coords=None,
                                                   point_labels=None,
                                                   box=bounding_boxes[None, :],
                                                   multimask_output=False)

        mask_imgs = []
        mask_color = np.array([30/255, 144/255, 255/255, 0.6])
        for mask in masks:
            h, w = mask.shape[-2:]
            mask = mask.astype(np.uint8)
            mask_img = mask.reshape(h, w, 1) * mask_color.reshape(1, 1, -1)
            mask_imgs.append(mask_img)

        return masks, mask_imgs


    # WORKING: YOLO_ASSIST:
    # find bounding boxes using YOLO
    # crop that part of the image and
    # send to a segmentation algorithm
    # can use opencv, scikit-image, ...
    # TODO: improve watershed, test canny edge
    def get_bounding_boxes_yolo(self, model: detect.ModelWrapper, img: str, confidence: float = 50.0, overlap: float = 50.0, use_slice = False, slice_wh=(640, 640), slice_overlap_ratio=(0.1, 0.1)):
        detection = detect.detect_objects(img_path=img,
                                            model=model,
                                            confidence=confidence,
                                            overlap=overlap,
                                            slice_detect=use_slice,
                                            slice_wh=slice_wh,
                                            slice_overlap_ratio=slice_overlap_ratio)
        
        # DEBUG
        imagetools.save_image_detection(img, "gn"+os.path.basename(img), save_dir="gn_test", detections=detection)

        boxes = detection.xyxy
        return boxes.astype("int32")

    def generate_crops(self, boxes: np.ndarray, img_path: str, save=False, save_preffix="crop"):
        img = cv2.imread(img_path)
        
        count = 0
        crops = []
        for box in boxes:
            top_left = (box[0], box[1])
            bottom_right = (box[2], box[3])

            crop = imagetools.crop_box(img, bottom_right, top_left)
            crops.append(crop)
            if save:
                imagetools.save_image(crop, f"{save_preffix}{count}{os.path.basename(img_path)}", dir="gn_test/crops")

            count += 1

        return crops


    def watershed(self, img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # gaussian blur first?
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        # Binary treshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # noise removal
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # erosion to improve foreground extraction
        kernel = np.ones((5,5), np.uint8)
        erosion = cv2.erode(thresh, kernel, iterations = 1)

        _, sure_fg = cv2.threshold(erosion, 0.2, 1.0, cv2.THRESH_BINARY)
        sure_bg = cv2.dilate(thresh, kernel, iterations=3)

        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1 # add 1 so that the sure background is 1
        markers[unknown == 255] = 0 # mark unknown region with 0
        markers = cv2.watershed(img, markers)

        marked_img = img.copy()
        # mark boundaries in red
        marked_img[markers == -1] = [255, 0, 0]
        return marked_img

    def canny_edge(self, img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gaussian blur first?
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        # Binary treshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        edges = cv2.Canny(thresh, threshold1=100, threshold2=200)
        return edges


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("img_src", 
                        nargs='*', 
                        default=None,
                        help="Source of image(s). Can be a single file, a list, or a directory")
    
    parser.add_argument("--method",
                        default=GenMethod.SAM2_YOLO_SEGMENT.value,
                        choices=list(range(1, len(GenMethod)+1)),
                        help=f"Method to generate. Default is (2) SAM2_YOLO_SEGMENT. Possible values are: {list(GenMethod)}")

    return parser.parse_args()

def load_model(path: str = f"{_GN_ROOT_PATH}/pretrained_models/train1/best.pt"):
    model = detect.ModelLoader("yolo").load(path=path)
    return model

def main():
    args = parse_args()

    img_src = args.img_src
    if img_src is None:
        raise RuntimeError("img_src is required")
    
    img_files = utils.get_all_files_from_paths(img_src)
    if len(img_files) == 0:
        raise RuntimeError(f"Couldn't retrieve any files from {img_src}")


    method = GenMethod(args.method)
    gn = Generator(method=method)
    match method:
        case GenMethod.SAM2_YOLO_SEGMENT:
            model = load_model()
            bounding_boxes = gn.get_bounding_boxes_yolo(model, img_files[0])

            masks, marked_imgs = gn.sam2_segment(img_path=img_files[0], bounding_boxes=bounding_boxes)
            plt.figure(figsize=(10,10))
            plt.imshow(np.array(Image.open(img_files[0]).convert("RGB")))
            for i, img in enumerate(marked_imgs):
                plt.imshow(img)
            plt.show()
        
        case GenMethod.YOLO_ASSIST:
            model = load_model()
            bounding_boxes = gn.get_bounding_boxes_yolo(model, img_files[0])
            print(bounding_boxes)

            crops = gn.generate_crops(bounding_boxes, img_files[0], save=True)
            for i, crop in enumerate(crops):
                marked = gn.watershed(crop)
                imagetools.save_image(marked, f"crop_wts{i}.png", dir="gn_test/wts")

                edges = gn.canny_edge(crop)
                imagetools.save_image(edges, f"crop_edges{i}.png", dir="gn_test/edge")

    gn.load_sam2()
    
if __name__ == "__main__":
    main()
        
