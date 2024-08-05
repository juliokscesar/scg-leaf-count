from modelloader import ModelLoader, ModelWrapper
import imagetools

img = "./imgs/onibus.jpeg"
model_path = "./pretrained_models/yolov8n.pt"

model = ModelLoader("yolo").load(path=model_path)
detections = model.predict(img)

annotated = imagetools.generate_annotated_image(img, detections)
imagetools.plot_image(annotated)
imagetools.save_image(annotated, "bus.jpeg")
