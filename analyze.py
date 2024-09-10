import argparse
import supervision as sv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import cv2
import os
from pathlib import Path

from scg_detection_tools.utils.file_handling import(
        read_yaml, get_all_files_from_paths, read_cached_detections
)
import scg_detection_tools.utils.image_tools as imtools
import scg_detection_tools.utils.cvt as cvt
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

    pd_parser.add_argument("method", choices=["segments", "boxes"], help="Method to base pixel density calculation. Segments will first use detection boxes and then segment them, while boxes will only use the deteciton boxes")
    
    pd_parser.add_argument("--on-slice",
                           dest="pd_on_slice",
                           action="store_true",
                           help="Calculate pixels by segmenting each object in each sliced detection, summing it all after finishing detection.")
    pd_parser.add_argument("--cached-detections",
                           dest="cached_detections",
                           type=str,
                           help="Path to directory containing .detections files of the images")
    pd_parser.add_argument("--segment-annotations",
                           dest="seg_annotations",
                           type=str,
                           help="Path to directory containing .txt files of segmententation annotations")
    pd_parser.add_argument("--show",
                           action="store_true",
                           help="Show pixel density plot")
    pd_parser.add_argument("--save-plot",
                           dest="save_plot",
                           action="store_true",
                           help="Save pixel density plot")
    
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


def analyze_pixel_density(model,
                          detector,
                          imgs,
                          sam2_ckpt_path=None,
                          sam2_cfg=None,
                          segments=False,
                          boxes=False,
                          slice_detection=False,
                          on_slice=False,
                          seg_annotations=None,
                          cached_detections=None,
                          show=True,
                          save=False):
    from analysis.pixel_density import pixel_density

    if not segments and not boxes:
        raise ValueError("Either 'segments' or 'boxes' must be True to choose method of pixel density calculation")
    
    densities = []

    cached_bboxes = None
    if cached_detections is not None:
        cached_bboxes = read_cached_detections(imgs, cached_detections)

    if segments:
        if seg_annotations is not None:
            ann_files, img_ann_idx = parse_seg_annotations(imgs, seg_annotations)
            for img in imgs:
                ann_file = ann_files[img_ann_idx[img]]
                _, contours = read_dataset_annotation(ann_file)

                imgsz = cv2.imread(img).shape[:2]
                contours_masks = cvt.contours_to_masks(contours, imgsz=imgsz)

                densities.append( pixel_density(imgsz=imgsz, masks=contours_masks) )
        
        else:
            from scg_detection_tools.segment import SAM2Segment
            
            if sam2_ckpt_path is None or sam2_cfg is None:
                    raise ValueError("sam2_ckpt_path and sam2_cfg are required in order to calculate pixel density on segments")

            seg = SAM2Segment(sam2_ckpt_path=sam2_ckpt_path,
                              sam2_cfg=sam2_cfg,
                              detection_assist_model=model)
            if on_slice:
                from analysis.geometry import mask_pixels
                for img in imgs:
                    total_pixels = cv2.imread(img).size // 3
                    result = seg.slice_segment_detect(img, (640,640))
                    
                    masks_pixels = 0
                    for slice in result["slices"]:
                        for mask in slice["masks"]:
                            masks_pixels += mask_pixels(mask)

                    densities.append( masks_pixels / total_pixels )
                    del result # save some memory
            
            else:
                if cached_detections is not None:
                    for img in imgs:
                        masks = seg._segment_boxes(img, boxes=cached_bboxes[img])
                        densities.append( pixel_density(img=img, masks=masks) )

                for img in imgs:
                    masks, _ = seg.detect_segment(img, slice_detection)
                    densities.append( pixel_density(img=img, masks=masks) )

        
    ## On Boxes
    else:
        if cached_detections is not None:
            for img in imgs:
                densities.append( pixel_density(img=img, boxes=cached_bboxes[img]) )
        else:
            detections = detector(imgs)
            for img, det in zip(imgs, detections):
                densities.append( pixel_density(img=img, boxes=det.xyxy.astype(np.int32)) )

    if save or show:
        fig, ax = plt.subplots(layout="constrained")
        img_ids = np.arange(1, len(imgs)+1)
        ax.plot(img_ids, densities, marker='o')
        ax.set(xlabel="Image ID", ylabel="Pixel density (objects_pixels / image_pixels)")

        if save:
            fig.savefig("exp_analysis/plots/pixel_density.png")
        if show:
            plt.show()

    save_to_csv(out_file="pixel_density.csv", img_id=img_ids, density=densities)
    return densities


def old_analyze_pixel_density(model, detector, imgs, cfg, on_slice=False, on_detection_boxes=False, cached_det_boxes=None, on_crops=False, seg_annotations=None, save_detections=False, show=False, save_plot=False):
    from analysis.pixel_density import pixel_density, slice_pixel_density, pixel_density_masks, pixel_density_boxes
    from scg_detection_tools.segment import SAM2Segment
    
    seg = SAM2Segment(sam2_ckpt_path=cfg["sam2_ckpt_path"],
                      sam2_cfg=cfg["sam2_cfg"],
                      detection_assist_model=model)
    if on_slice:
        

        results, densities = slice_pixel_density(img_files=imgs, slice_wh=(640,640), seg=seg)
        detections = [result["detections"] for result in results]

    elif seg_annotations:
        from scg_detection_tools.utils.cvt import contours_to_masks
        ann_files, img_ann_idx = parse_seg_annotations(imgs, seg_annotations)
        ann_masks = []
        for img in imgs:
            ann_file = ann_files[img_ann_idx[img]]
            _, contours = read_dataset_annotation(ann_file)
            
            imgsz = cv2.imread(img).shape[:2]
            contours_masks = contours_to_masks(contours, imgsz=imgsz)
            ann_masks.append(contours_masks)

        densities = pixel_density_masks(imgs=imgs, imgs_masks=ann_masks)

    elif on_detection_boxes:
        if cached_det_boxes is not None:
            from scg_detection_tools.utils.cvt import boxes_to_masks

            bboxes = read_cached_detections(imgs, cached_det_boxes)
            if len(bboxes) == 0:
                raise RuntimeError(f"No cached detections found in {cached_det_boxes}")
            imgs_boxes = [bboxes[img] for img in imgs]

            densities = pixel_density_boxes(imgs=imgs, imgs_boxes=imgs_boxes)
                
        else:
            detections = detector.detect_objects(imgs)
            densities = pixel_density(imgs=imgs, detections=detections, save_img_masks=True, seg=seg)

    elif on_crops:
        detections = detector.detect_objects(imgs)
        densities = pixel_density(imgs=imgs, detections=detections, on_crops=on_crops, seg=seg)

    else:
        raise UserWarning("Either on_slice, on_detection_boxes or seg_annotations must have a true value to calculate pixel density")

    if save_detections:
        for img, detection in zip(imgs,detections):
            imtools.save_image_detection(default_imgpath=img, detections=detection, save_name=f"pd_det{os.path.basename(img)}", save_dir="exp_analysis")

    print(f"Calculated pixel density for each image: {[(img,density) for img,density in zip(imgs, densities)]}")
    img_ids = np.arange(1, len(imgs)+1)
    save_to_csv(out_file="pixel_density.csv", img_ids=img_ids, densities=densities)
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(img_ids, densities)
    if show:
        plt.show()
    if save_plot:
        fig.savefig("exp_analysis/plots/pixel_density.png")
    return densities


def analyze_color_histogram(model, detector, imgs, raw=False, on_detection_boxes=False, seg_annotations=None, cspaces=["RGB"], show=True, save_plots=False):
    from analysis.color_analysis import color_hist
    from scg_detection_tools.utils.cvt import contours_to_masks, boxes_to_masks

    if seg_annotations is not None:
        ann_files, img_ann_idx = parse_seg_annotations(imgs, seg_annotations)

    if save_plots and not os.path.isdir("exp_analysis/plots"):
        os.makedirs("exp_analysis/plots")

    img_hists = {}
    for img in imgs:
        hists = None
        img_masks = None
        img_size = cv2.imread(img).shape[:2]
        if raw:
            hists = color_hist(img, cspaces=cspaces)
            
        elif on_detection_boxes:
            detections = detector(img)[0]
            boxes = detections.xyxy.astype(np.int32)
            img_masks = boxes_to_masks(boxes=boxes, imgsz=img_size)
            mask_classes = np.zeros(len(img_masks))
            hists = color_hist(img, cspaces=cspaces, boxes=boxes)

        elif seg_annotations:
            ann_file = ann_files[img_ann_idx[img]]
            mask_classes, contours = read_dataset_annotation(ann_file)
            hists = color_hist(img, cspaces=cspaces, masks_contours=contours, mask_classes=mask_classes)
            img_masks = contours_to_masks(contours=contours, imgsz=img_size)
            
            
            
        img_hists[img] = hists
        if show:
            for cspace in cspaces:
                full_hist = hists[cspace]["full"]
                img_data = cv2.imread(img)
                orig = img_data.copy()

                fig, axs = plt.subplots(nrows=1, ncols=2, layout="constrained", figsize=(12,8))
                axs[0].axis("off")
                
                if cspace == "GRAY":
                    ch_labels = ["GRAY"]
                    ch_colors = ["k"]
                    ch_cmap = "gray"
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
                elif cspace == "HSV":
                    ch_labels = ["H", "S", "V"]
                    ch_colors = ["r", "g", "b"]
                    ch_cmap = "hsv"
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2HSV)
                elif cspace == "RGB":
                    ch_labels = ["R", "G", "B"]
                    ch_colors = ["r", "g", "b"]
                    ch_cmap = "viridis"
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

                
                for i in range(len(full_hist)):
                    ch_hist = full_hist[i]
                    axs[1].plot(ch_hist, color=ch_colors[i], label=ch_labels[i])
                axs[1].legend()
                axs[1].set(xlabel=f"{cspace} Intensity",
                           ylabel="Frequency")
    
                if img_masks is not None:
                    masks_hists = hists[cspace]["masks"]
                    for mask,mask_class in zip(img_masks, mask_classes):
                        if mask_class == 0:
                            color = [30, 6, 255]
                        elif mask_class == 1:
                            color = [30, 255, 6]
                        elif mask_class == 2:
                            color = [255, 6, 6]
                        else:
                            color = [4, 4, 4]
                        alpha = 0.8

                        img_data = imtools.segment_annotated_image(img_data, mask, color=color, alpha=alpha)

                axs[0].imshow(img_data, cmap=ch_cmap)
                if save_plots:
                    fig.savefig(f"exp_analysis/hist_{cspace}_{os.path.basename(img)}")

                plt.show()


    return img_hists


def analyze_classify(detector, 
                     imgs,
                     cls_labels, 
                     cls_colors: dict,
                     method: str = "knn",
                     seg_annotations: str = None,
                     sam2_ckpt_path: str = None,
                     sam2_cfg: str = None,
                     save=False):
    
    from analysis.classify import KNNClassifier, SVMClassifier

    # Check if will use pre-annotated masks or SAM2
    if seg_annotations is not None:
        ann_files, img_ann_idx = parse_seg_annotations(imgs, seg_annotations)
    elif sam2_cfg is not None or sam2_ckpt_path is not None:
        from scg_detection_tools.segment import SAM2Segment
        seg = SAM2Segment(sam2_ckpt_path=sam2_ckpt_path, sam2_cfg=sam2_cfg)
    else:
        raise ValueError("Either 'seg_annotations' or SAM2 details must be provided.")

    # Get classifier(s) based on method
    method = method.strip().lower()
    if method == "knn":
        clf = KNNClassifier(n_neighbors=5)
        clf.load_state("knn_k5_last.skl")
    elif method == "svm":
        clf = SVMClassifier(kernel="rbf")
        clf.load_state("svm_rbf_last.skl")
    elif method == "nn":
        raise NotImplemented()
    else:
        raise ValueError(f"Method {method} is not valid. Possible options are: 'knn', 'svm', 'nn'")


    # Prepare color patches for legend when showing results
    color_patches = [
        mpatches.Patch(color=cls_colors[c], label=c) for c in cls_colors
    ]


    # Get every detected object mask
    image_objects = {}
    for img in imgs:
        imgsz = cv2.imread(img).shape[:2]
        if seg_annotations is not None:
            ann_file = ann_files[img_ann_idx[img]]
            _, mask_contours = read_dataset_annotation(ann_file)
            masks = cvt.contours_to_masks(mask_contours, imgsz=imgsz)
            boxes = []
            for contour in mask_contours:
                c_points = np.array(contour).reshape(len(contour)//2, 2)
                box = cvt.segment_to_box(c_points, normalized=True, imgsz=imgsz)
                boxes.append(box)

            
        elif seg is not None:
            detections = detector(img)[0]
            masks = seg._segment_detection(img, detections)
            boxes = detections.xyxy.astype(np.int32)
        

        image_objects[img] = (masks, boxes)


    # Now apply mask to segment our object and crop a box around it
    OBJ_STD_SIZE = (32,32)
    for img, (masks, boxes) in image_objects.items():
        orig_img = cv2.imread(img)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        # First, keep track of every object mask, box and a crop around it
        obj_data = []
        
        for box, mask in zip(boxes, masks):
            h, w = mask.shape[1:]
            mask = mask.astype(np.uint8).reshape(h, w,)
            mask = np.where(mask == 1, 255, 0)
            
            masked = orig_img.copy()
            masked[mask[:,:] < 255] = 0

            obj_crop = imtools.crop_box_image(masked, box)
            obj_crop = cv2.resize(obj_crop, OBJ_STD_SIZE, cv2.INTER_CUBIC)

            obj_data.append( [box, mask, obj_crop] )

        # Prepare our input to the classifier,
        # make X vector containing the object's
        # rgb, hsv and gray pixels
        # then classify using this vector

        obj_cls = {}
        cls_count = {l: 0 for l in cls_labels}
        for data_idx, (box, mask, obj) in enumerate(obj_data):
            hsv = cv2.cvtColor(obj, cv2.COLOR_RGB2HSV)
            gray = cv2.cvtColor(obj, cv2.COLOR_RGB2GRAY)
            rgb = obj

            attributes = np.concatenate((rgb.flatten(), hsv.flatten(), gray.flatten()))
            nclass = clf.predict([attributes])[0]

            label = cls_labels[nclass]
            cls_count[label] += 1
            obj_cls[data_idx] = nclass

        # Count every class occurrence and plot image
        # with mask annotations
        fig, axs = plt.subplots(ncols=2, figsize=(15,10))
        axs[0].axis("off")
        axs[0].imshow(orig_img)

        axs[1].axis("off")

        ann_img = orig_img.copy()
        for data_idx in obj_cls:
            nclass = obj_cls[data_idx]
            _, mask, _ = obj_data[data_idx]

            class_label = cls_labels[nclass]
            color = []
            if isinstance(cls_colors[class_label], str):
                color.extend(mcolors.to_rgb(cls_colors[class_label]))
            else:
                color.extend(cls_colors[class_label])
            
            ann_img = imtools.segment_annotated_image(ann_img, mask, color, alpha=0.6)

        axs[1].imshow(ann_img)
        axs[1].legend(handles=color_patches)

        plt.show()

        print(f"Total objects: {len(obj_data)}")
        for l in cls_labels:
            print(f"Class {l!r}: {cls_count[l]}")



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
        # analyze_pixel_density(model, det, img_files, cfg=cfg, on_slice=args.pd_slice, on_detection_boxes=args.on_detection_boxes, cached_det_boxes=args.cached_detections, seg_annotations=args.seg_annotations, save_detections=args.save_detections, show=args.show, save_plot=args.save_plot)
        method = args.method
        analyze_pixel_density(model, det, img_files, sam2_ckpt_path=cfg["sam2_ckpt_path"], sam2_cfg=cfg["sam2_cfg"], 
                              boxes=(method == "boxes"), segments=(method == "segments"), slice_detection=det_params["use_slice"],
                              on_slice=args.pd_on_slice, seg_annotations=args.seg_annotations, cached_detections=args.cached_detections,
                              show=args.show, save=args.save_plot)

    elif args.command == "color_hist":
        analyze_color_histogram(model, det, img_files, raw=args.raw, on_detection_boxes=args.detection_boxes, seg_annotations=args.seg_annotations)
    


if __name__ == "__main__":
    main()

