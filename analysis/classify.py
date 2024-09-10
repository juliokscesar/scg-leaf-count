from abc import ABC, abstractmethod
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn import svm
import matplotlib.pyplot as plt
import joblib

from scg_detection_tools.utils.file_handling import file_exists

class BaseClassifier(ABC):
    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        pass

    @abstractmethod
    def load_state(self, state_file: str):
        pass

    @abstractmethod
    def save_state(self, file_name: str):
        pass


class KNNClassifier(BaseClassifier):
    def __init__(self, n_neighbors: int, weights: str = "distance", state_file: str = None):
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

    def train(self, X_train, y_train):
        self._clf.fit(X_train, y_train)

    def predict(self, X):
        return self._clf.predict(X)


    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        cm = confusion_matrix(y_test, predictions, labels=self._clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        print(classification_report(y_test, predictions, labels=self._clf.classes_))
        disp.plot()
        plt.show()

    
    def load_state(self, state_file: str):
        if not file_exists(state_file):
            raise FileExistsError(f"State file {state_file} does not exist")

        with open(state_file, "rb") as sf:
            self._clf = joblib.load(sf)


    def save_state(self, file_name: str):
        with open(file_name, "wb") as sf:
            joblib.dump(self._clf, sf)


class SVMClassifier(BaseClassifier):
    def __init__(self, kernel="rbf", state_file: str = None):
        if state_file is not None:
            self.load_state(state_file)
            return

        self._clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("svc", svm.SVC(gamma="auto", kernel=kernel))
            ]
        )


    def train(self, X_train, y_train):
        self._clf.fit(X_train, y_train)


    def predict(self, X):
        return self._clf.predict(X)


    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        cm = confusion_matrix(y_test, predictions, labels=self._clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        print(classification_report(y_test, predictions, labels=self._clf.classes_))
        disp.plot()
        plt.show()

    
    def load_state(self, state_file: str):
        if not file_exists(state_file):
            raise FileExistsError(f"State file {state_file} does not exist")

        with open(state_file, "rb") as sf:
            self._clf = joblib.load(sf)


    def save_state(self, file_name: str):
        with open(file_name, "wb") as sf:
            joblib.dump(self._clf, sf)

