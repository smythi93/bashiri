import enum
import os
from abc import ABC
from typing import Collection, Sequence, Tuple, Any, Optional, List

import pandas as pd
import shap
from joblib import dump, load
from sflkit.runners.run import TestResult
from sklearn import svm, tree
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

from bashiri.features import FeatureVector, Feature
from bashiri.reduce import FeatureSelection, DefaultSelection


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
        self.all_features: Sequence[Feature] = list()
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

    def train(self):
        self.model.fit(
            self.reducer.select(self.x_train).to_numpy(), self.y_train.to_numpy()
        )
        if self.path:
            dump(self.model, str(self.path))

    def fit(
        self,
        all_features: Sequence[Feature],
        feature_vectors: Collection[FeatureVector],
    ):
        self.all_features = all_features
        self.x_train, self.y_train = self.prepare_data(all_features, feature_vectors)
        self.train()

    def evaluate(
        self,
        feature_vectors: Collection[FeatureVector],
        output_dict: bool = False,
    ) -> Tuple[str | dict, List[List[int]]]:
        x_eval, y_eval = self.prepare_data(self.all_features, feature_vectors)
        return (
            self.classification_report(x_eval, y_eval, output_dict=output_dict),
            confusion_matrix(
                y_eval.to_numpy(), self.model.predict(x_eval.to_numpy())
            ).tolist(),
        )

    def classify(self, x: pd.DataFrame) -> Label:
        return Label[int(self.model.predict(x.to_numpy())[0])]

    def accuracy_score(self, x: pd.DataFrame, y: pd.DataFrame):
        return accuracy_score(y.to_numpy(), self.model.predict(x.to_numpy()))

    def classification_report(
        self, x: pd.DataFrame, y: pd.DataFrame, output_dict: bool = False
    ) -> str | dict:
        return classification_report(
            y.to_numpy(), self.model.predict(x.to_numpy()), output_dict=output_dict
        )

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

    def finalize(
        self,
        feature_vectors: Collection[FeatureVector],
    ):
        x_train, y_train = self.prepare_data(self.all_features, feature_vectors)
        self.x_train = pd.concat((self.x_train, x_train))
        self.y_train = pd.concat((self.y_train, y_train))
        self.train()


class SVMOracle(Oracle):
    def __init__(
        self,
        path: Optional[os.PathLike] = None,
        reducer: FeatureSelection = DefaultSelection(),
    ):
        if path is None or not os.path.exists(path):
            model = svm.SVC()
        else:
            model = load(str(path))
        super().__init__(model=model, path=path, reducer=reducer)


class SGDOracle(Oracle):
    def __init__(
        self,
        path: Optional[os.PathLike] = None,
        reducer: FeatureSelection = DefaultSelection(),
        loss: str = "log_loss",
    ):
        if path is None or not os.path.exists(path):
            model = SGDClassifier(loss=loss, penalty="l1")
        else:
            model = load(str(path))
        super().__init__(model=model, path=path, reducer=reducer)


class NeuralNetworkOracle(Oracle):
    def __init__(
        self,
        path: Optional[os.PathLike] = None,
        reducer: FeatureSelection = DefaultSelection(),
        random_state: int = 42,
        solver: str = "lbfgs",
    ):
        # solver: lbfgs, sgd, or adam
        if path is None or not os.path.exists(path):
            model = MLPClassifier(random_state=random_state, solver=solver)
        else:
            model = load(str(path))
        super().__init__(model=model, path=path, reducer=reducer)


class DecisionTreeOracle(Oracle):
    def __init__(
        self,
        path: Optional[os.PathLike] = None,
        reducer: FeatureSelection = DefaultSelection(),
        random_state: int = 42,
        criterion: str = "log_loss",
    ):
        # criterion: gini or log_loss
        if path is None or not os.path.exists(path):
            model = tree.DecisionTreeClassifier(
                random_state=random_state, criterion=criterion
            )
        else:
            model = load(str(path))
        super().__init__(
            model=model, path=path, reducer=reducer, explainer=shap.TreeExplainer
        )
