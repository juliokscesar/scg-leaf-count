from ultralytics import YOLO
from roboflow import Roboflow
import supervision as sv
import numpy as np
from pathlib import Path
import cv2

import utils

def load_yolo_model(path: str) -> YOLO:
    return YOLO(path)


def load_roboflow_model(api_key: str, project: str, version: int):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project(project)
    model = project.version(version).model

    return model



MODEL_TYPE_LOADING_FUNCS = {
    "yolo": load_yolo_model,
    "roboflow": load_roboflow_model
}

SUPPORTED_MODEL_TYPES = ["yolo", "roboflow"]


class ModelWrapper:
    def __init__(self, model_type: str, underlying_model: any):
        if (model_type not in SUPPORTED_MODEL_TYPES):
            raise "model_type must be one of: yolo, roboflow"
        
        
        self._model_type = model_type
        self._underlying_model = underlying_model


    def predict(self, img_path: str, confidence: float = 50.0, overlap: float = 50.0) -> sv.Detections:
        match self._model_type:
            case "yolo":
                results = self._underlying_model.predict(img_path, imgsz=640, conf=confidence / 100.0, iou=overlap / 100.0)
                detections = sv.Detections.from_ultralytics(results[0])
            
            case "roboflow":
                results = self._underlying_model.predict(img_path, confidence=confidence, overlap=overlap).json()
                detections = sv.Detections.from_inference(results)

            case _:
                return None

        return detections
    

    def slice_predict(self, img_path: str, confidence: float, overlap: float, slice_wh=(640, 640), slice_overlap_ratio=(0.1, 0.1)) -> sv.Detections:
        def sv_slice_callback(image: np.ndarray) -> sv.Detections:
            tmpfile = utils.generate_temp_path(Path(img_path).suffix)
            with open(tmpfile, "wb") as f:
                cv2.imwrite(f.name, image)

                return self.predict(f.name, confidence=confidence, overlap=overlap)
            
        image  = cv2.imread(img_path)

        slicer = sv.InferenceSlicer(callback=sv_slice_callback, slice_wh=slice_wh, overlap_ratio_wh=slice_overlap_ratio)
        sliced_detections = slicer(image=image)

        utils.clear_temp_folder()
        return sliced_detections


class ModelLoader:
    def __init__(self, model_type: str):
        if (model_type not in SUPPORTED_MODEL_TYPES):
            raise "model_type must be one of: yolo, roboflow"
        
        self._model_type = model_type

        self._loading_func = MODEL_TYPE_LOADING_FUNCS[model_type]

    
    def load(self, **kwargs) -> ModelWrapper:
        match self._model_type:
            case "yolo":
                if not utils.ensure_arg_in_kwargs(kwargs, "path"):
                    raise "Must have 'path' keyword argument when loading YOLO model"
                
            case "roboflow":
                if not utils.ensure_arg_in_kwargs(kwargs, "api_key", "project", "version"):
                    raise "Must have 'api_key', 'project', and 'version' keyword arguments when loading Roboflow model"
                
        model = self._loading_func(**kwargs)

        return ModelWrapper(self._model_type, model)
    