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
import torch.optim as optim
from torchvision import models, transforms
import numpy as np
from copy import deepcopy


from scg_detection_tools.utils.file_handling import file_exists

class BaseClassifier(ABC):
    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def fit(self, X_train, y_train):
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

    def fit(self, X_train, y_train):
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


    def fit(self, X_train, y_train):
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

    def fit(self, X_train, y_train):
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


class MLPClassifier(nn.Module, BaseClassifier):
    def __init__(self, n_features: int, n_classes: int, preprocess=None, state_file=None):
        if state_file is not None:
            self.load_state(state_file)
            return
        
        super().__init__()

        self.fc1 = nn.Linear(n_features, n_features//2)
        self.bn1 = nn.BatchNorm1d(n_features//2)
        self.ac1 = nn.ReLU()
        self.fc2 = nn.Linear(n_features//2, n_features//4)
        self.bn2 = nn.BatchNorm1d(n_features//4)
        self.ac2 = nn.ReLU()
        self.fc3 = nn.Linear(n_features//4, n_classes)

        self._preprocess = preprocess
        device = "cuda" if torch.cuda.is_available() else "cpu"
        nn.Module.to(self, device)
        self._device = device

    @staticmethod
    def from_state(state_file: str):
        with open(state_file, "rb") as f:
            obj = dill.load(f)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        obj._device = device
        nn.Module.to(obj, device)
        obj.eval()
        return obj

    def forward(self, X):
        X = self.fc1(X)
        X = self.bn1(X)
        X = self.ac1(X)

        X = self.fc2(X)
        X = self.bn2(X)
        X = self.ac2(X)

        X = self.fc3(X)
        return X

    def fit(self, X_train, y_train, epochs=20):
        if self._preprocess is not None:
            X_train = self._preprocess(X_train)

        X = torch.tensor(deepcopy(X_train)).to(self._device)
        labels = torch.tensor(deepcopy(y_train)).to(self._device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        correct = 0
        total = 0

        hist_loss = []
        hist_acc = []
        self.train() # Set to training mode (from nn.Module)
        for epoch in range(epochs):
            optimizer.zero_grad() # Zero the gradients

            # Forward pass and compute loss
            outputs = self(X)
            loss = criterion(outputs, labels)

            # Backpropagate errors to adjust parameters
            loss.backward() # Backpropagate
            optimizer.step() # Update weights

            _, predicted =  torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100.0 * correct / total

            hist_loss.append(loss.item())
            hist_acc.append(accuracy)

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

        _, axs = plt.subplots(nrows=1, ncols=2, figsize=(14,8))
        axs[0].set(xlabel="Epoch", ylabel="Loss", title="Loss")
        axs[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
        ep = np.arange(1, epochs+1)
        axs[0].plot(ep, hist_loss)
        axs[1].plot(ep, hist_acc)
        plt.show()

    
    def predict(self, X):
        if self._preprocess is not None:
            X = self._preprocess(X)
        trans = torch.tensor(deepcopy(X)).to(self._device)

        with torch.no_grad():
            output = self(trans)

        _, predicted = torch.max(output, 1)
        return predicted.cpu().numpy()


    def evaluate(self, X_test, y_test, disp_labels=None):
        self.eval()

        predictions = self.predict(X_test)
        cm = confusion_matrix(y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
        print(classification_report(y_test, predictions, target_names=disp_labels))
        disp.plot()
        plt.show()


######################################################################
g_RESNET_INSTANCES = { 
    18: None,
    34: None,
    50: None,
}
g_RESNET_PREPROCESS = None
### Function to extract feature from images using ResNet
# TODO: implement option resnets instead of having to change it here everytime
def resnet_extract_features(img: np.ndarray, resnet: int = 18):
    if resnet not in g_RESNET_INSTANCES:
        raise ValueError("'resnet' must be either 18, 34 or 50")
    if (g_RESNET_INSTANCES[resnet] is None) or (g_RESNET_PREPROCESS is None):
        _init_resnet(resnet)
    
    proc = g_RESNET_PREPROCESS(img).unsqueeze(0)
    with torch.no_grad():
        features = g_RESNET_INSTANCE[resnet](proc)
    return features.squeeze()

def _init_resnet(which: int = 18):
    global g_RESNET_INSTANCE, g_RESNET_PREPROCESS

    if which == 18:
        g_RESNET_INSTANCE[which] = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif which == 34:
        g_RESNET_INSTANCE[which] = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    else:
        g_RESNET_INSTANCE[which] = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    g_RESNET_INSTANCE[which] = nn.Sequential(*list(g_RESNET_INSTANCE[which].children())[:-1])
    for p in g_RESNET_INSTANCE[which].parameters():
        p.requires_grad = False

    g_RESNET_INSTANCE[which].eval()

    if g_RESNET_PREPROCESS is None:
        g_RESNET_PREPROCESS = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
