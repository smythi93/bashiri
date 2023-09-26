import enum
from abc import abstractmethod, ABC
from typing import Dict, List, Set, Tuple

from sflkit.analysis.analysis_type import AnalysisObject, EvaluationResult
from sflkit.analysis.factory import CombinationFactory, analysis_factory_mapping
from sflkit.analysis.spectra import Spectrum
from sflkit.analysis.predicate import Predicate
from sflkit.model import Scope, EventFile, Model
from sflkit.runners.run import TestResult
from sflkitlib.events.event import Event


class FeatureValue(enum.Enum):
    TRUE = 1
    FALSE = 0
    UNDEFINED = -1

    def __repr__(self):
        return self.name

    def __or__(self, other):
        if isinstance(other, Feature):
            if other == FeatureValue.TRUE or self == FeatureValue.UNDEFINED:
                return other
            else:
                return self
        elif isinstance(other, bool):
            if other:
                return FeatureValue.TRUE
            elif self == FeatureValue.UNDEFINED:
                if other is None:
                    return FeatureValue.UNDEFINED
                else:
                    return FeatureValue.FALSE
            else:
                return self
        else:
            return self


class Feature(ABC):
    def __init__(self, name: str, analysis: AnalysisObject):
        self.name = name
        self.analysis = analysis

    @abstractmethod
    def default(self):
        raise NotImplementedError()

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return hasattr(other, "name") and self.name == other.name

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.__repr__()

    def __lt__(self, other):
        if hasattr(other, "name"):
            return self.name < other.name
        else:
            raise TypeError(
                f"'<' not supported between instances of '{type(self)}' and '{type(other)}'"
            )


class BinaryFeature(Feature):
    def default(self):
        return FeatureValue.FALSE


class TertiaryFeature(Feature):
    def default(self):
        return FeatureValue.UNDEFINED


class FeatureVector:
    def __init__(
        self,
        result: TestResult,
    ):
        self.result = result
        self.features: Dict[Feature, FeatureValue] = dict()

    def get_feature_value(self, feature: Feature) -> FeatureValue:
        if feature in self.features:
            return self.features[feature]
        else:
            return feature.default()

    def set_feature(self, feature: Feature, value: FeatureValue):
        if feature not in self.features:
            self.features[feature] = value
        else:
            self.features[feature] = self.features[feature] or value

    def get_features(self) -> Dict[Feature, FeatureValue]:
        return dict(self.features)

    def vector(self, features: List[Feature]) -> List[FeatureValue]:
        return [self.get_feature_value(feature) for feature in features]

    def num_vector(self, features: List[Feature]) -> List[int]:
        return [value.value for value in self.vector(features)]

    def dict_vector(self, features: List[Feature]) -> Dict[Feature, FeatureValue]:
        return {feature: self.get_feature_value(feature) for feature in features}

    def num_dict_vector(self, features: List[Feature]) -> Dict[str, int]:
        return {
            feature.name: value.value
            for feature, value in self.dict_vector(features).items()
        }

    def tuple(self, features: List[Feature]) -> Tuple[Tuple[Feature, FeatureValue]]:
        return tuple((feature, self.get_feature_value(feature)) for feature in features)

    def __repr__(self):
        return f"{self.result.name}{self.features}"

    def __str__(self):
        return f"{self.result.name}{self.features}"

    def __eq__(self, other):
        if isinstance(other, FeatureVector) and self.result == other.result:
            for feature in set(self.features.keys()).union(set(other.features.keys())):
                if self.get_feature_value(feature) != other.get_feature_value(feature):
                    return False
            return True
        else:
            return False

    def difference(self, other, features: List[Feature]):
        if isinstance(other, FeatureVector):
            s = 0
            for feature in features:
                s += self.get_feature_value(feature) != other.get_feature_value(feature)
            return s
        else:
            return 0


class FeatureBuilder(CombinationFactory):
    def __iter__(self) -> FeatureVector:
        yield from self.feature_vectors.values()

    def __next__(self) -> FeatureVector:
        yield from self.feature_vectors.values()

    def __len__(self):
        return len(self.feature_vectors)

    def run_ids(self):
        return set(self.feature_vectors.keys())

    def get_vector_by_id(self, run_id: int):
        return self.feature_vectors.get(run_id, None)

    def get_vectors(self) -> List[FeatureVector]:
        return list(self.feature_vectors.values())

    def get_all_features(self) -> List[Feature]:
        return sorted(list(self.all_features))

    def remove(self, run_id: int):
        if run_id in self.feature_vectors:
            del self.feature_vectors[run_id]

    def __init__(self):
        super().__init__(list(map(lambda f: f(), analysis_factory_mapping.values())))
        self.analysis: List[AnalysisObject] = list()
        self.feature_vectors: Dict[int, FeatureVector] = dict()
        self.all_features: Set[Feature] = set()

    def get_analysis(self, event, scope: Scope = None) -> List[AnalysisObject]:
        self.analysis = super().get_analysis(event, scope)
        self.analysis.append(self)
        return self.analysis

    @staticmethod
    def map_evaluation(analysis: Spectrum, id_: int):
        match analysis.get_last_evaluation(id_):
            case EvaluationResult.TRUE:
                return FeatureValue.TRUE
            case EvaluationResult.FALSE:
                return FeatureValue.FALSE
            case True:
                return FeatureValue.TRUE
            case False:
                return FeatureValue.FALSE
            case _:
                return FeatureValue.UNDEFINED

    def prepare(self, run_id: int, test_result: TestResult):
        self.feature_vectors[run_id] = FeatureVector(test_result)

    def hit(self, id_, *args, **kwargs):
        event: Event
        for a in self.analysis:
            if isinstance(a, Spectrum):
                feature = BinaryFeature(str(a), a)
                self.feature_vectors[id_].set_feature(
                    feature, self.map_evaluation(a, id_)
                )
                self.all_features.add(feature)
            elif isinstance(a, Predicate):
                feature = TertiaryFeature(str(a), a)
                self.feature_vectors[id_].set_feature(
                    feature, self.map_evaluation(a, id_)
                )
                self.all_features.add(feature)

    def copy(self):
        new_feature_builder = FeatureBuilder()
        new_feature_builder.all_features = set(self.all_features)
        new_feature_builder.feature_vectors = dict(self.feature_vectors)
        return new_feature_builder


class EventHandler:
    def __init__(self):
        self.feature_builder = FeatureBuilder()
        self.model = Model(self.feature_builder)

    @staticmethod
    def map_result(failing: bool):
        match failing:
            case True:
                return TestResult.FAILING
            case False:
                return TestResult.PASSING
            case _:
                return TestResult.UNDEFINED

    def handle(self, event_file: EventFile):
        self.model.prepare(event_file.run_id)
        self.feature_builder.prepare(
            event_file.run_id, self.map_result(event_file.failing)
        )
        with event_file:
            for event in event_file.load():
                event.handle(self.model)

    def handle_files(self, event_files: List[EventFile]):
        for e in event_files:
            self.handle(e)

    def copy(self):
        new_handler = EventHandler()
        new_handler.feature_builder = self.feature_builder.copy()
        new_handler.model = Model(new_handler.feature_builder)
        return new_handler
