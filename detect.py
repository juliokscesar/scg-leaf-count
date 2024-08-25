from modelloader import ModelWrapper, ModelLoader, SUPPORTED_MODEL_TYPES
import utils
import imagetools
import supervision as sv
import argparse
import os

def detect_objects(img_path: str, 
                    model: ModelWrapper, 
                    confidence: float = 50,
                    overlap: float = 50,
                    slice_detect=False,
                    slice_wh: tuple[int, int] | None = None,
                    slice_overlap_ratio: tuple[float, float] | None = None,
                    embed_slice_callback=None) -> sv.Detections:
    
    if slice_detect:
        if slice_wh is None or slice_overlap_ratio is None:
                raise Exception("Slice detection requires slice_wh (tuple[int]) and slice_overlap_ratio (tuple[float]) arguments")
        
        detections = model.slice_predict(img_path=img_path, 
                                         confidence=confidence, 
                                         overlap=overlap, 
                                         slice_wh=slice_wh, 
                                         slice_overlap_ratio=slice_overlap_ratio,
                                         embed_slice_callback=embed_slice_callback)

    else:
        detections = model.predict(img_path=img_path, confidence=confidence, overlap=overlap)

    return detections


def count_objects(detections: sv.Detections) -> int:
    return len(detections.xyxy)

def count_in_image(
        img_path: str, 
        model: ModelWrapper, 
        confidence: float = 50.0,
        overlap: float = 50.0,
        slice_detect=False,
        slice_wh: tuple[int, int] | None = None,
        slice_overlap_ratio: tuple[float, float] | None = None) -> int:
    
    detections = detect_objects(img_path, model, confidence, overlap, slice_detect, slice_wh, slice_overlap_ratio)
    return count_objects(detections)



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("img_path", help="Path to image")

    parser.add_argument("--model-type", 
                        choices=SUPPORTED_MODEL_TYPES, 
                        default="yolo", dest="model_t", 
                        help=f"Model Type. Default is 'yolo'. Possible options are: {', '.join(SUPPORTED_MODEL_TYPES)}")
    parser.add_argument("--model-path", 
                        default="yolov8n.pt",
                        dest="model_path",
                        help="Path to model. Default uses yolov8n.pt")
    
    parser.add_argument("-c", 
                        "--confidence", 
                        type=float, 
                        default="50.0", 
                        help="Confidence parameter. Default is 50.0")
    parser.add_argument("-o", 
                        "--overlap", 
                        type=float, 
                        default="50.0", 
                        help="Overlap parameter. Default is 50.0")
    
    parser.add_argument("-s", 
                        "--slice-detect", 
                        action="store_true", 
                        dest="slice",
                        help="Use slice detection")
    parser.add_argument("--slice-w", 
                        type=int, 
                        dest="slice_w", 
                        default=640,
                        help="Slice width when using slice detection. Default is 640")
    parser.add_argument("--slice-h", 
                        type=int, 
                        dest="slice_h", 
                        default=640,
                        help="Slice height when using slice detection. Default is 640")
    parser.add_argument("--slice-overlap", 
                        type=float, 
                        dest="slice_overlap", 
                        default=50.0,
                        help="Slice overlap ratio when using slice detection. Default is 50.0")

    parser.add_argument("--save", action="store_true", help="Save image with detections")
    parser.add_argument("--no-show", action="store_true", dest="no_show", help="Don't plot image with detections")

    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    img_path = args.img_path
    model_t = args.model_t
    model_path = args.model_path
    conf = args.confidence
    overlap = args.overlap

    use_slice = args.slice
    slice_wh = (int(args.slice_w), int(args.slice_h))
    slice_overlap = float(args.slice_overlap)

    save_image = args.save
    no_show = args.no_show

    if not utils.file_exists(img_path):
        print(f"File does not exist: {img_path}")

    if model_t == "yolo":
        model = ModelLoader(model_t).load(path=model_path)

    elif model_t == "yolonas":
        model = ModdelLoader(model_t).load(model_arch="yolo_nas_l",
                                           num_classes=1,
                                           checkpoint_path=model_path)

    elif model_t == "roboflow":
        API_KEY = os.environ["ROBOFLOW_API_KEY"]

        project = input("Roboflow project: ")
        version = int(input("Roboflow project version: "))

        model = ModelLoader(model_t).load(api_key=API_KEY, project=project, version=version)

    else:
        raise Exception(f"model-type must be one of: {', '.join(SUPPORTED_MODEL_TYPES)}")

    
    detections = detect_objects(img_path=img_path, 
                                model=model,
                                confidence=conf,
                                overlap=overlap,
                                slice_detect=use_slice,
                                slice_wh=slice_wh,
                                slice_overlap_ratio=(slice_overlap/100.0, slice_overlap/100.0))

    print(f"Counted: {count_objects(detections)}")
    
    if not no_show:
        imagetools.plot_image_detection(img_path, detections, box_thickness=2)
    if save_image:
        imagetools.save_image_detection(default_imgpath=img_path, save_name="detect"+os.path.basename(img_path), save_dir="exp", detections=detections)

if __name__ == "__main__":
    main()
