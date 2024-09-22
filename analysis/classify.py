from abc import ABC, abstractmethod
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import dill
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np


from scg_detection_tools.utils.file_handling import file_exists

class BaseClassifier(ABC):
    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test, disp_labels=None):
        pass

    @staticmethod
    def from_state(state_file: str):
        with open(state_file, "rb") as f:
            obj = dill.load(f)
        return obj

    
    def save_state(self, file_name: str):
        with open(file_name, "wb") as f:
            dill.dump(self, f)


class KNNClassifier(BaseClassifier):
    def __init__(self, n_neighbors: int, weights: str = "distance", enable_nca = False, state_file: str = None, preprocess=None):
        if state_file is not None:
            self.load_state(state_file)
            return
        if enable_nca:
            self._clf = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("nca", NeighborhoodComponentsAnalysis(random_state=42)),
                    ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)),
                ]
            )
        else:
            self._clf = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)),
                ]
            )
        self._n_neighbors = n_neighbors
        self._preprocess = preprocess

    def train(self, X_train, y_train):
        if self._preprocess is not None:
            X_train = self._preprocess(X_train)
        self._clf.fit(X_train, y_train)


    def predict(self, X):
        if self._preprocess is not None:
            X = self._preprocess(X)
        return self._clf.predict(X)


    def evaluate(self, X_test, y_test, disp_labels=None):
        predictions = self.predict(X_test)
        cm = confusion_matrix(y_test, predictions, labels=self._clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
        print(classification_report(y_test, predictions, labels=self._clf.classes_, target_names=disp_labels))
        disp.plot()
        plt.show()


class SVMClassifier(BaseClassifier):
    def __init__(self, kernel="rbf", state_file: str = None, preprocess = None):
        if state_file is not None:
            self.load_state(state_file)
            return
        self._clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("svc", svm.SVC(gamma="auto", kernel=kernel))
            ]
        )
        self._preprocess = preprocess


    def train(self, X_train, y_train):
        if self._preprocess is not None:
            X_train = self._preprocess(X_train)
        self._clf.fit(X_train, y_train)


    def predict(self, X):
        if self._preprocess is not None:
            X = self._preprocess(X)
        return self._clf.predict(X)


    def evaluate(self, X_test, y_test, disp_labels=None):
        predictions = self.predict(X_test)
        cm = confusion_matrix(y_test, predictions, labels=self._clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
        print(classification_report(y_test, predictions, labels=self._clf.classes_, target_names=disp_labels))
        disp.plot()
        plt.show()


class SGDBasedClassifier(BaseClassifier):
    def __init__(self, loss="hinge", max_iter=1000,  preprocess=None, state_file=None):
        if state_file is not None:
            self.load_state(state_file)
            return
        self._clf = Pipeline(
            steps = [
                ("scaler", StandardScaler()),
                ("sgd", SGDClassifier(loss=loss, max_iter=max_iter)),
            ]
        )
        self._preprocess = preprocess

    def train(self, X_train, y_train):
        if self._preprocess is not None:
            X_train = self._preprocess(X_train)
        self._clf.fit(X_train, y_train)

    def predict(self, X):
        if self._preprocess is not None:
            X = self._preprocess(X)
        return self._clf.predict(X)
    
    def evaluate(self, X_test, y_test, disp_labels=None):
        predictions = self.predict(X_test)
        cm = confusion_matrix(y_test, predictions, labels=self._clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
        print(classification_report(y_test, predictions, labels=self._clf.classes_, target_names=disp_labels))
        disp.plot()
        plt.show()


######################################################################
g_RESNET_INSTANCE = None
g_RESNET_PREPROCESS = None
### Function to extract feature from images using ResNet
def resnet_extract_features(img: np.ndarray):
    if (g_RESNET_INSTANCE is None) or (g_RESNET_PREPROCESS is None):
        _init_resnet()
    
    proc = g_RESNET_PREPROCESS(img).unsqueeze(0)
    with torch.no_grad():
        features = g_RESNET_INSTANCE(proc)
    return features.squeeze()

def _init_resnet():
    global g_RESNET_INSTANCE, g_RESNET_PREPROCESS

    g_RESNET_INSTANCE = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    #_RESNET_INSTANCE = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    #_RESNET_INSTANCE = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    g_RESNET_INSTANCE = nn.Sequential(*list(g_RESNET_INSTANCE.children())[:-1])
    for p in g_RESNET_INSTANCE.parameters():
        p.requires_grad = False

    g_RESNET_INSTANCE.eval()

    g_RESNET_PREPROCESS = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
