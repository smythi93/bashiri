import enum
import os
from abc import ABC, abstractmethod
from typing import Collection, Sequence, Tuple, Any, Optional

import pandas as pd
import shap
from joblib import dump, load
from sflkit.runners.run import TestResult
from sklearn import svm, __all__, tree
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier

from stato.features import FeatureVector, Feature
from stato.reduce import FeatureSelection, DefaultSelection


class Label(enum.Enum):
    BUG = 1
    NO_BUG = 0


class Oracle(ABC):
    def __init__(
        self,
        model: Any,
        path: Optional[os.PathLike] = None,
        reducer: FeatureSelection = DefaultSelection(),
        explainer=shap.KernelExplainer,
    ):
        self.model = model
        self.path = path
        self.reducer = reducer
        self.explainer = explainer
        self.x_train: pd.DataFrame = None
        self.y_train: pd.DataFrame = None

    @staticmethod
    def prepare_data(
        all_features: Sequence[Feature],
        feature_vectors: Collection[FeatureVector],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data = list()
        for feature_vector in feature_vectors:
            num_dict = feature_vector.num_dict_vector(all_features)
            num_dict["label"] = 1 if feature_vector.result == TestResult.FAILING else 0
            data.append(num_dict)
        data = pd.DataFrame(data)
        return data.drop(columns="label"), data["label"]

    def train(self, x_train: pd.DataFrame, y_train: pd.DataFrame):
        self.model.fit(x_train.to_numpy, y_train.to_numpy)
        if self.path:
            dump(self.model, str(self.path))

    def fit(
        self,
        all_features: Sequence[Feature],
        feature_vectors: Collection[FeatureVector],
    ):
        self.x_train, self.y_train = self.prepare_data(all_features, feature_vectors)
        self.train(self.reducer.select(self.x_train), self.y_train)

    def classify(self, x: pd.DataFrame) -> Label:
        return Label[int(self.model.predict(x.to_numpy())[0])]

    def evaluate(self, x: pd.DataFrame, y: pd.DataFrame):
        return accuracy_score(y.to_numpy(), self.model.predict(x.to_numpy()))

    def report(self, x: pd.DataFrame, y: pd.DataFrame) -> str | dict:
        return classification_report(y.to_numpy(), self.model.predict(x.to_numpy()))

    def explain(self, x: pd.DataFrame, out: Optional[os.PathLike] = None):
        x = x.to_numpy()
        explainer = self.explainer(self.model, x)
        shap_values = explainer.shap_values(x)
        try:
            plot = shap.summary_plot(shap_values, x, show=out is not None)
        except:
            print("shap.summary_plot() failed")
        else:
            if out is not None:
                shap.save_html(out, plot)

    def improve(self, x: pd.DataFrame, y: pd.DataFrame):
        self.fit(self.x_train.join(x), self.y_train.join(y))

    def finalize(self):
        self.fit(self.x_train, self.y_train)


class SVM(Oracle):
    def __init__(
        self,
        path: Optional[os.PathLike] = None,
        reducer: FeatureSelection = DefaultSelection(),
    ):
        if path is None:
            model = load(str(path))
        else:
            model = svm.SVC()
        super().__init__(model=model, path=path, reducer=reducer)


class SGD(Oracle):
    def __init__(
        self,
        path: Optional[os.PathLike] = None,
        reducer: FeatureSelection = DefaultSelection(),
        loss: str = "log_loss",
    ):
        if path is None:
            model = load(str(path))
        else:
            model = SGDClassifier(loss=loss, penalty="l1")
        super().__init__(model=model, path=path, reducer=reducer)


class NeuralNetwork(Oracle):
    def __init__(
        self,
        path: Optional[os.PathLike] = None,
        reducer: FeatureSelection = DefaultSelection(),
        random_state: int = 42,
        solver: str = "lbfgs",
    ):
        # solver: lbfgs, sgd, or adam
        if path is None:
            model = load(str(path))
        else:
            model = MLPClassifier(random_state=random_state, solver=solver)
        super().__init__(model=model, path=path, reducer=reducer)


class DecisionTree(Oracle):
    def __init__(
        self,
        path: Optional[os.PathLike] = None,
        reducer: FeatureSelection = DefaultSelection(),
        random_state: int = 42,
        criterion: str = "log_loss",
    ):
        # criterion: gini or log_loss
        if path is None:
            model = load(str(path))
        else:
            model = tree.DecisionTreeClassifier(
                random_state=random_state, criterion=criterion
            )
        super().__init__(
            model=model, path=path, reducer=reducer, explainer=shap.TreeExplainer
        )
