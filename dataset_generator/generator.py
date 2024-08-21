import os
import sys
import shutil
from pathlib import Path

_GN_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
sys.path.append(_GN_ROOT_PATH)

from enum import Enum
import numpy as np
import cv2
import torch
import sam2
import sam2.build_sam
import sam2.sam2_image_predictor
import argparse
import yaml
import math

import detect
import utils
import imagetools


class GenMethod(Enum):
    SAM2_YOLO_SEGMENT = 1
    YOLO_ASSIST = 2
    MANUAL_SEGMENT = 3

    def __str__(self):
        return self.name

class Generator:
    def __init__(self, 
                 method: GenMethod = GenMethod.YOLO_ASSIST,
                 yolo_model_path: str = f"{_GN_ROOT_PATH}/pretrained_models/train1/best.pt",
                 **kwargs):
        self._method = method

        self._yolo_model = detect.ModelLoader("yolo").load(path=yolo_model_path)

        self._sam2_predictor = None
        if self._method == GenMethod.SAM2_YOLO_SEGMENT:
            if "sam2_chkpt_path" not in kwargs: 
                self.load_sam2()
            else:
                self.load_sam2(sam2_chkpt_path=kwargs["sam2_chkpt_path"])

            self._sam2_on_slice_buffer = {}

    def __call__(self):
        pass

    # TODO: Manual segment:
    # use different individual and group images of leafs
    # segment them manually
    # load the image and the segmentation data
    # randomize a lot of them in a background image
    # keep track of every segmentation transformation,
    # including translation, rotation
    def gen_from_samples(self,
                         backgrounds_path: str = f"{_GN_ROOT_PATH}/dataset_generator/samples/backgrounds",
                         individual_path: str = f"{_GN_ROOT_PATH}/dataset_generator/samples/individual",
                         max_images: int = 30,
                         out_path: str = f"{_GN_ROOT_PATH}/gn_manual_dataset/"):
        
        bg_imgs = imagetools.load_imgs(backgrounds_path)
        
        ind_imgs = {}
        # search for (o)riginal and (s)caled
        files = utils.get_all_files_from_paths(individual_path)
        ind_imgs["original"] = imagetools.load_imgs(list(filter(lambda f: f[0]=='o', files)))
        ind_imgs["scaled"] = imagetools.load_imgs(list(filter(lambda f: f[0]=='s', files)))

        

    
    ##############################################################################
    ############# SAM2 USAGE FUNCTIONS
    ##############################################################################
    def load_sam2(self, sam2_chkpt_path: str = f"{_GN_ROOT_PATH}/dataset_generator/sam2chkpts/sam2_hiera_tiny.pt", sam2_cfg: str = "sam2_hiera_t.yaml"):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        sam2_model = sam2.build_sam.build_sam2(sam2_cfg, sam2_chkpt_path, device=str(device))
        predictor = sam2.sam2_image_predictor.SAM2ImagePredictor(sam2_model)
        self._sam2_predictor = predictor

    def sam2_contours_from_masks(self, masks: np.ndarray):
        mask_contours = []
        for mask in masks:
            h, w = mask.shape[-2:]
            mask = mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask.reshape(h,w,1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_contours.append(contours)

        # mask_contours comes out as an array of contours
        # contours comes as an array containing an array
        # which contains every point, but the points are in another array as well
        # so this is some magic to have mask_contours[i]=ithcontour,
        # and ithcontour contains every point of that contour
        fmt_contours = []
        for contours in mask_contours:
            for contour in contours:
                fmt_contours.append([])
                for points in contour:
                    fmt_contours[-1].append(points[0])

        mask_contours = fmt_contours
        return mask_contours

    def sam2_img_with_masks(self, img, masks: np.ndarray)-> np.ndarray:
        if isinstance(img, str):
            dest = cv2.imread(img)
            dest = cv2.cvtColor(dest, cv2.COLOR_BGR2RGBA)
        else:
            dest = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
            dest[:,:,3] = 1.0


        mask_color = np.array([30/255, 144/255, 255/255, 1.0])
        for mask in masks:
            h, w = mask.shape[-2:]
            mask = mask.astype(np.uint8)
            mask_img = mask.reshape(h, w, 1) * mask_color.reshape(1, 1, -1)

            alpha_mask = mask_img[:,:,3]
            alpha_dest = 1.0 - alpha_mask

            for c in range(3):
                dest[:,:, c] = (alpha_mask * mask_img[:,:,c] + alpha_dest * dest[:,:,c])

        return dest

    def sam2_segment(self, img_path: str, bounding_boxes: np.ndarray):
        if self._sam2_predictor is None:
            raise RuntimeError("Need to load sam2 model first")
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self._sam2_predictor.set_image(img)
        masks, _, _ = self._sam2_predictor.predict(point_coords=None,
                                                   point_labels=None,
                                                   box=bounding_boxes[None, :],
                                                   multimask_output=False)

        return masks

    def sam2_segment_slices(self, img_path: str):
        full_bboxes = self.get_bounding_boxes_yolo(img=img_path,
                                                   use_slice=True,
                                                   embed_slice_callback=self.sam2_on_slice_callback)
        
        self._sam2_on_slice_buffer[img_path]["bboxes"] = full_bboxes

        return self._sam2_on_slice_buffer[img_path]

    def sam2_segment_oncrops(self, crops: np.ndarray):
        if self._sam2_predictor is None:
            raise RuntimeError("Need to load sam2 model first")

        masks_imgs = []
        for crop in crops:
            self._sam2_predictor.set_image(crop)
            h, w, _ = crop.shape
            mid_point = np.array([[w//2, h//2]])

            masks, _, _ = self._sam2_predictor.predict(point_coords=mid_point,
                                                       point_labels=np.array([1]),
                                                       multimask_output=False) 

            mask_crop = self.sam2_img_with_masks(crop, masks)
            masks_imgs.append((masks, mask_crop))
            

        return masks_imgs


    def sam2_on_slice_callback(self, img_path: str, img: np.ndarray, tmp_slice_path: str, bboxes: np.ndarray):
        if img_path not in self._sam2_on_slice_buffer:
            self._sam2_on_slice_buffer[img_path] = {"slices": []}

        if len(bboxes) == 0:
            return

        slice_buffer = { 
            "orig": img,
            "bboxes": bboxes,
            "masks": None,
            "masks_img": None,
            "slice_save_path": ""
        }

        masks = self.sam2_segment(tmp_slice_path, bboxes)
        slice_buffer["masks"] = masks
        slice_buffer["masks_img"] = self.sam2_img_with_masks(img, masks)
        
        save_name = f"sam2mk_{len(self._sam2_on_slice_buffer[img_path]['slices'])}_{os.path.basename(img_path)}"
        save_dir = f"{_GN_ROOT_PATH}/dataset_generator/gn_cache/sam2slice"
        imagetools.save_image(img, save_name, dir=save_dir, convert_to_BGR=True)
        slice_buffer["slice_save_path"] = f"{save_dir}/{save_name}"

        self._sam2_on_slice_buffer[img_path]["slices"].append(slice_buffer)


    def get_bounding_boxes_yolo(self, 
                                img: str, 
                                confidence: float = 50.0, 
                                overlap: float = 50.0, 
                                use_slice = False, 
                                slice_wh=(640, 640), 
                                slice_overlap_ratio=(0.1, 0.1),
                                embed_slice_callback=None):
        detection = detect.detect_objects(img_path=img,
                                            model=self._yolo_model,
                                            confidence=confidence,
                                            overlap=overlap,
                                            slice_detect=use_slice,
                                            slice_wh=slice_wh,
                                            slice_overlap_ratio=slice_overlap_ratio,
                                            embed_slice_callback=embed_slice_callback)
        
        # DEBUG
        imagetools.save_image_detection(img, "gn"+os.path.basename(img), save_dir="gn_test", detections=detection)

        boxes = detection.xyxy
        return boxes.astype(np.int32)

    ##############################################################################
    ##############################################################################
    ##############################################################################

    def generate_crops(self, boxes: np.ndarray, img_path: str, save=False, save_preffix="crop"):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
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

        return markers

    def canny_edge(self, img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        # Binary treshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        edges = cv2.Canny(thresh, threshold1=100, threshold2=200)
        return edges

    def extract_contours(self, marks: np.ndarray):
        contours, _ = cv2.findContours(marks.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours

    ##############################################################################


    ##############################################################################
    ####### DATASET FILE HANDLING (READ, WRITE, MASK TRANSFORMATIONS, ETC)
    ##############################################################################
    
    def write_to_dataset(self, orig_img: str, contours: np.ndarray, out_file: str = "gendata.yaml", out_dir: str = "gn_dataset"):
        img = cv2.imread(orig_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        if not os.path.exists(f"{_GN_ROOT_PATH}/{out_dir}"):
            for t in ["train", "valid", "test"]:
                os.makedirs(f"{_GN_ROOT_PATH}/{out_dir}/{t}/images")
                os.makedirs(f"{_GN_ROOT_PATH}/{out_dir}/{t}/labels")

        with open(f"{_GN_ROOT_PATH}/{out_dir}/train/labels/{Path(orig_img).stem}.txt", "w") as f:
            for contour in contours:
                contour_line = f"0"
                # dataset format needs normalized (W,H)
                for point in contour:
                    norm = [point[0] / w, point[1] / h]
                    contour_line += f" {norm[0]} {norm[1]}"

                f.write(contour_line + '\n')

        shutil.copy(orig_img, f"{_GN_ROOT_PATH}/{out_dir}/train/images")

        if not utils.file_exists(f"{_GN_ROOT_PATH}/{out_dir}/{out_file}"):
            dataset = {
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "nc": 1,
                "names": ["leaf"]
            }
            with open(f"{_GN_ROOT_PATH}/{out_dir}/{out_file}", "w") as f:
                yaml.dump(dataset, f, default_flow_style=False)

    ##############################################################################

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("img_src", 
                        nargs='*', 
                        default=None,
                        help="Source of image(s). Can be a single file, a list, or a directory")
    
    method_gp = parser.add_mutually_exclusive_group()
    method_gp.add_argument("--yolo-assist", dest="yolo_assist", action="store_true")
    method_gp.add_argument("--sam2-yolo-segment", dest="sam2_yolo", action="store_true", default=True)
    method_gp.add_argument("--manual-segment", dest="manual_segment", action="store_true")

    parser.add_argument("--yolo-slice", dest="yolo_slice", action="store_true", help="Use slice detection for yolo assisted methods.")

    parser.add_argument("--only-crop", dest="only_crop", action="store_true", help="Just crop boxes and save crops images")
    parser.add_argument("--only-image", dest="only_img", action="store_true", help="Only generate images marked with detections, but down create the dataset.")

    return parser.parse_args()

def load_model(path: str = f"{_GN_ROOT_PATH}/pretrained_models/train1/best.pt"):
    model = detect.ModelLoader("yolo").load(path=path)
    return model

def main():
    args = parse_args()

    img_src = args.img_src
    if img_src is None:
        raise RuntimeError("img_src is required")
    
    img_files = utils.get_all_files_from_paths(*img_src)
    if len(img_files) == 0:
        raise RuntimeError(f"Couldn't retrieve any files from {img_src}")


    if args.yolo_assist:
        method = GenMethod.YOLO_ASSIST
    elif args.sam2_yolo:
        method = GenMethod.SAM2_YOLO_SEGMENT
    elif args.manual_segment:
        method = GenMethod.MANUAL_SEGMENT
    else:
        raise Exception("method needs to have a value")
    
    use_slice = args.yolo_slice
    only_crop = args.only_crop
    only_img = args.only_img

    gn = Generator(method=method)

    if only_crop:
        gn.generate_crops(bounding_boxes, img_files[0], save=True)
        return

    print(img_files)

    if method == GenMethod.SAM2_YOLO_SEGMENT:
        for img_file in img_files:
            img = cv2.imread(img_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            marked_imgs = []

            h, w = img.shape[:2]
            if h > 640 and w > 640:
                result = gn.sam2_segment_slices(img_file)
                slc = result["slices"]

                marked_imgs = [s["masks_img"] for s in slc]
                masks = [s["masks"] for s in slc]
                slice_paths = [s["slice_save_path"] for s in slc]

                for s,m in zip(slice_paths, masks):
                    contours = gn.sam2_contours_from_masks(m)
                    if not only_img:
                        gn.write_to_dataset(s, contours, out_file="sam2segdata.yaml", out_dir="gn_sam2segdataset")

            else:
                bboxes = gn.get_bounding_boxes_yolo(img_file)
                result = gn.sam2_segment(img_file, bboxes)
                marked_img = gn.sam2_img_with_masks(img_file, result)
                marked_imgs.append(marked_img.copy())

                contours = gn.sam2_contours_from_masks(result)
                if not only_img:
                    gn.write_to_dataset(img_file, contours, "sam2segdata.yaml", "gn_sam2segdataset")
                

            for i in range(len(marked_imgs)):
                imagetools.save_image(marked_imgs[i], f"sam2mk_{i}_{os.path.basename(img_file)}", f"{_GN_ROOT_PATH}/dataset_generator/gn_cache/sam2/marked", convert_to_BGR=True)

    elif method == GenMethod.YOLO_ASSIST:
        bounding_boxes = gn.get_bounding_boxes_yolo(img_files[0], use_slice=use_slice)

        crops = gn.generate_crops(bounding_boxes, img_files[0], save=True)
        for i, crop in enumerate(crops):
            marked = gn.watershed(crop)
            imagetools.save_image(marked, f"crop_wts{i}.png", dir="gn_test/wts")

            edges = gn.canny_edge(crop)
            imagetools.save_image(edges, f"crop_edges{i}.png", dir="gn_test/edge")




    # match method:
    #     case GenMethod.SAM2_YOLO_SEGMENT:
    #         for img_file in img_files:
    #             img = cv2.imread(img_file)
    #             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #             marked_imgs = []

    #             h, w = img.shape[:2]
    #             if h > 640 and w > 640:
    #                 result = gn.sam2_segment_slices(img_file)
    #                 slc = result["slices"]

    #                 marked_imgs = [s["masks_img"] for s in slc]
    #                 masks = [s["masks"] for s in slc]
    #                 slice_paths = [s["slice_save_path"] for s in slc]

    #                 for s,m in zip(slice_paths, masks):
    #                     contours = gn.sam2_contours_from_masks(m)
    #                     if not only_img:
    #                         gn.write_to_dataset(s, contours, out_file="sam2segdata.yaml", out_dir="gn_sam2segdataset")

    #             else:
    #                 bboxes = gn.get_bounding_boxes_yolo(img_file)
    #                 result = gn.sam2_segment(img_file, bboxes)
    #                 marked_img = gn.sam2_img_with_masks(img_file, result)
    #                 marked_imgs.append(marked_img.copy())

    #                 contours = gn.sam2_contours_from_masks(result)
    #                 if not only_img:
    #                     gn.write_to_dataset(img_file, contours, "sam2segdata.yaml", "gn_sam2segdataset")
                    

    #             for i in range(len(marked_imgs)):
    #                 imagetools.save_image(marked_imgs[i], f"sam2mk_{i}_{os.path.basename(img_file)}", f"{_GN_ROOT_PATH}/dataset_generator/gn_cache/sam2/marked", convert_to_BGR=True)

    #     case GenMethod.YOLO_ASSIST:
    #         bounding_boxes = gn.get_bounding_boxes_yolo(img_files[0], use_slice=use_slice)

    #         crops = gn.generate_crops(bounding_boxes, img_files[0], save=True)
    #         for i, crop in enumerate(crops):
    #             marked = gn.watershed(crop)
    #             imagetools.save_image(marked, f"crop_wts{i}.png", dir="gn_test/wts")

    #             edges = gn.canny_edge(crop)
    #             imagetools.save_image(edges, f"crop_edges{i}.png", dir="gn_test/edge")


    
if __name__ == "__main__":
    main()
        
