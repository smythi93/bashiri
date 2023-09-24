from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class FeatureSelection(ABC):
    def __init__(self):
        self.length = 0

    def prerequisite(self, features: pd.DataFrame):
        assert len(features) >= 1, "At least on feature vector required"
        assert (
            len(set(map(len, features))) == 1
        ), "All feature vectors must have the same length"
        self.length = len(features[0])

    def select(self, features: pd.DataFrame) -> pd.DataFrame:
        self.prerequisite(features)
        return self.choices(features)

    @abstractmethod
    def choices(self, features: pd.DataFrame) -> pd.DataFrame:
        pass


class DefaultSelection(FeatureSelection):
    def prerequisite(self, features: pd.DataFrame):
        pass

    def choices(self, features: pd.DataFrame) -> pd.DataFrame:
        return features


class RemoveIrrelevantFeatures(FeatureSelection):
    def choices(self, features: pd.DataFrame) -> List[List[int]]:
        drop = []
        for col in features.columns:
            if len(set(features[col].to_list())) <= 1:
                drop.append(col)
        return features.drop(columns=drop)
