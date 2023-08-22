from abc import ABC, abstractmethod
from typing import List


class FeatureSelection(ABC):
    def __init__(self):
        self.length = 0

    def prerequisite(self, features: List[List[int]]):
        assert len(features) >= 1, "At least on feature vector required"
        assert (
            len(set(map(len, features))) == 1
        ), "All feature vectors must have the same length"
        self.length = len(features[0])

    def select(self, features: List[List[int]]) -> List[List[int]]:
        self.prerequisite(features)
        return self.choices(features)

    @abstractmethod
    def choices(self, features: List[List[int]]) -> List[List[int]]:
        pass


class RemoveIrrelevantFeatures(FeatureSelection):
    def choices(self, features: List[List[int]]) -> List[List[int]]:
        selection = [list() for _ in features]
        for i in range(self.length):
            current_feature = list(map(lambda f: f[i], features))
            if len(set(current_feature)) > 1:
                for s, f in zip(selection, current_feature):
                    s.append(f)
        return selection
