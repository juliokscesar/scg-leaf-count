from abc import ABC, abstractmethod
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn import svm
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
    def __init__(self, n_neighbors: int, weights: str = "distance", state_file: str = None, preprocess=None):
        if state_file is not None:
            self.load_state(state_file)
            return

        self._clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
            #   ("nca", NeighborhoodComponentsAnalysis(random_state=42)),
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
        print(classification_report(y_test, predictions, labels=self._clf.classes_))
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
        print(classification_report(y_test, predictions, labels=self._clf.classes_))
        disp.plot()
        plt.show()




######################################################################
### Function to extract feature from images using ResNet
_RESNET_INSTANCE = None
_RESNET_PREPROCESS = None
def resnet_extract_features(img: np.ndarray):
    if (_RESNET_INSTANCE is None) or (_RESNET_PREPROCESS is None):
        _init_resnet()
    
    proc = _RESNET_PREPROCESS(img).unsqueeze(0)
    with torch.no_grad():
        features = _RESNET_INSTANCE(proc)
    return features.squeeze()

def _init_resnet():
    global _RESNET_INSTANCE, _RESNET_PREPROCESS

    _RESNET_INSTANCE = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    _RESNET_INSTANCE = nn.Sequential(*list(_RESNET_INSTANCE.children())[:-1])
    for p in _RESNET_INSTANCE.parameters():
        p.requires_grad = False

    _RESNET_INSTANCE.eval()

    _RESNET_PREPROCESS = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
