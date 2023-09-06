import enum
from abc import ABC, abstractmethod
from typing import Collection, Sequence, Tuple

import pandas as pd
from sflkit.runners.run import TestResult

from stato.features import FeatureVector, Feature
from stato.reduce import FeatureSelection, DefaultSelection


class Label(enum.Enum):
    BUG = 1
    NO_BUG = 0


class Oracle(ABC):
    def __init__(self, reducer: FeatureSelection = DefaultSelection()):
        self.reducer = reducer
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

    @abstractmethod
    def train(self, x_train: pd.DataFrame, y_train: pd.DataFrame):
        pass

    def fit(
        self,
        all_features: Sequence[Feature],
        feature_vectors: Collection[FeatureVector],
    ):
        self.x_train, self.y_train = self.prepare_data(all_features, feature_vectors)
        self.train(self.reducer.select(self.x_train), self.y_train)

    @abstractmethod
    def classify(self, x: pd.DataFrame) -> Label:
        return Label.NO_BUG

    def improve(self, x: pd.DataFrame, y: pd.DataFrame):
        self.fit(self.x_train.join(x), self.y_train.join(y))

    def finalize(self):
        self.fit(self.x_train, self.y_train)
