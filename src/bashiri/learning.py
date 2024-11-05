import enum
import os
from abc import ABC
from typing import Sequence, Tuple, Any, Optional, List

import pandas as pd
import shap
from joblib import dump, load
from sflkit.features.handler import EventHandler
from sflkit.features.value import Feature
from sklearn import svm, tree
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

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

    def prepare_data(
        self,
        handler: EventHandler,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data = handler.to_df(self.all_features)
        return data.drop(columns=["test", "failing"]), data["failing"]

    def train(self):
        self.model.fit(
            self.reducer.select(self.x_train).to_numpy(), self.y_train.to_numpy()
        )
        if self.path:
            dump(self.model, str(self.path))

    def fit(
        self,
        all_features: Sequence[Feature],
        handler: EventHandler,
    ):
        self.all_features = all_features
        self.x_train, self.y_train = self.prepare_data(handler)
        self.train()

    def evaluate(
        self,
        handler: EventHandler,
        output_dict: bool = False,
    ) -> Tuple[str | dict, List[List[int]]]:
        x_eval, y_eval = self.prepare_data(handler)
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

    def explain(self, x: Optional[pd.DataFrame] = None):
        x = x or self.x_train
        return self.explainer(self.model, x)(x)

    def finalize(
        self,
        handler: EventHandler,
    ):
        x_train, y_train = self.prepare_data(handler)
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
