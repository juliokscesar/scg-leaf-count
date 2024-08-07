from modelloader import ModelLoader, ModelWrapper
import imagetools
import os

img = "./imgs/oldplant0.jpg"
model_path = "pretrained_models/yolov8s.pt"

yolo_model = ModelLoader("yolo").load(path=model_path)

RBF_API_KEY = os.environ["ROBOFLOW_API_KEY"]
RBF_PROJECT = "plant-leaf-detection-att1p"
RBF_VERSION = 2
rbf_model = ModelLoader("roboflow").load(api_key=RBF_API_KEY, project=RBF_PROJECT, version=RBF_VERSION)

detections = rbf_model.predict(img)
#detections = model.slice_predict(img, confidence=50.0, overlap=50.0, slice_wh=(640, 640))



annotated = imagetools.generate_annotated_image(img, detections)
imagetools.plot_image(annotated)
imagetools.save_image(annotated, "exp.jpeg")
