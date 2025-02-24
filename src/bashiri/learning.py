import enum
import os
import warnings
from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Any, Optional, List, Callable

import numpy as np
import pandas as pd
import shap
from causalml.inference.tree import CausalTreeRegressor, CausalRandomForestRegressor
from joblib import dump, load
from sflkit.analysis.predicate import Predicate
from sflkit.analysis.spectra import Spectrum
from sflkit.events.event_file import EventFile
from sflkit.features.handler import EventHandler
from sflkit.features.value import Feature
from sklearn import svm, tree
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

from bashiri.reduce import FeatureSelection, DefaultSelection

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


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


class Operation(enum.Enum):
    AND = "AND"
    OR = "OR"


class CausalFinder(ABC):
    def __init__(
        self,
        reducer: FeatureSelection = DefaultSelection(),
    ):
        self.reducer = reducer
        self.all_features: Sequence[Feature] = list()
        self.x_train: pd.DataFrame = None
        self.y_train: pd.DataFrame = None
        self.treatments: pd.DataFrame = None

    @abstractmethod
    def get_model(self):
        pass

    def prepare_data(
        self,
        handler: EventHandler,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data = handler.to_df(self.all_features)
        return data.drop(columns=["test", "failing"]), data["failing"]

    def train(self):
        model = self.get_model()
        model.fit(
            self.reducer.select(self.x_train).to_numpy(),
            self.treatments.to_numpy(),
            self.y_train.to_numpy(),
        )
        return model

    def prepare(
        self,
        all_features: Sequence[Feature],
        handler: EventHandler,
    ):
        self.all_features = all_features
        self.x_train, self.y_train = self.prepare_data(handler)

    def fit(
        self,
        treatments: pd.DataFrame,
    ):
        model = self.get_model()
        model.fit(
            self.reducer.select(self.x_train).to_numpy(),
            treatments.to_numpy(),
            self.y_train.to_numpy(),
        )
        return model


class CausalTree(CausalFinder):
    def __init__(
        self,
        reducer: FeatureSelection = DefaultSelection(),
        random_state: int = 42,
        criterion: str = "causal_mse",
    ):
        super().__init__(
            reducer=reducer,
        )
        self.random_state = random_state
        self.criterion = criterion

    def get_model(self):
        return CausalTreeRegressor(
            random_state=self.random_state, criterion=self.criterion
        )


class CausalForest(CausalFinder):
    def __init__(
        self,
        reducer: FeatureSelection = DefaultSelection(),
        random_state: int = 42,
        criterion: str = "causal_mse",
    ):
        super().__init__(
            reducer=reducer,
        )
        self.random_state = random_state
        self.criterion = criterion

    def get_model(self):
        return CausalRandomForestRegressor(
            random_state=self.random_state, criterion=self.criterion
        )


class Bashiri(DecisionTreeOracle):
    def __init__(
        self,
        handler: EventHandler,
        finder: CausalFinder,
        event_files: Sequence[EventFile],
        metric: Callable[[Spectrum | Predicate], float] = Spectrum.Tarantula,
        path: Optional[os.PathLike] = None,
        reducer: FeatureSelection = DefaultSelection(),
        random_state: int = 42,
        criterion: str = "log_loss",
        ns: List[int] = None,
        ops: List[Operation] = None,
    ):
        super().__init__(
            path=path, reducer=reducer, random_state=random_state, criterion=criterion
        )
        self.handler = handler
        self.all_features = handler.builder.get_all_features()
        self.data = handler.to_df(self.all_features)
        self.oracle = None
        self.metric = metric
        self.passing, self.failing = [e.run_id for e in event_files if not e.failing], [
            e.run_id for e in event_files if e.failing
        ]
        self.analysis: List[Spectrum | Predicate] = list(handler.builder.get_all())
        for a in self.analysis:
            a.finalize(self.passing, self.failing)
            a.assign_suspiciousness(metric)
        self.analysis.sort(reverse=True)
        self.ns = ns or [1, 2, 3, 5]
        self.ops = ops or [Operation.AND, Operation.OR]
        self.finder = finder
        self.finder.prepare(self.all_features, handler)
        self.treatments = dict()
        for n in self.ns:
            for op in self.ops:
                name, treatment = self.get_treatment(n, op)
                # check if multiple treatments
                if np.unique(treatment).shape[0] <= 1:
                    continue
                self.treatments[name] = self.finder.fit(treatment)

    def prepare_data(
        self,
        handler: EventHandler,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data = handler.to_df(self.all_features)
        x = data.drop(columns=["test", "failing"])
        return (
            pd.concat(
                [
                    x,
                    pd.DataFrame(
                        np.array(
                            [
                                self.treatments[treatment].predict(x.to_numpy())
                                for treatment in self.treatments
                            ]
                        ).T,
                        columns=list(self.treatments.keys()),
                    ),
                ],
                axis=1,
            ),
            data["failing"],
        )

    def get_treatment(self, n: int, op: Operation) -> Tuple[str, pd.DataFrame]:
        top_n = [str(a) for a in self.analysis[:n]]
        name = f"{op.value}_{'_'.join(top_n)}"
        if op == Operation.AND:
            return name, self.data[top_n].all(axis=1).astype(int)
        else:
            return name, self.data[top_n].any(axis=1).astype(int)
