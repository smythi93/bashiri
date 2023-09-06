from abc import ABC, abstractmethod
from typing import List, Dict

from sflkit.runners.run import TestResult

from stato.events import EventCollector
from stato.features import Handler, FeatureVector
from stato.learning import Oracle, Label


class FeedbackLoop(ABC):
    def __init__(
        self,
        handler: Handler,
        oracle: Oracle,
        seeds: Dict[int, List[str]],
        iterations: int = 10,
    ):
        self.handler = handler
        self.features = handler.feature_builder
        self.oracle = oracle
        self.seeds = seeds
        self.iterations = iterations
        self.all_features = self.features.all_features

    @abstractmethod
    def iteration(self):
        pass

    def run(self):
        for _ in range(self.iterations):
            self.iteration()
            self.all_features = self.features.all_features
        self.oracle.finalize()


class TestGenFeedback(FeedbackLoop, ABC):
    def __init__(
        self,
        handler: Handler,
        oracle: Oracle,
        seeds: Dict[int, List[str]],
        collector: EventCollector,
        iterations: int = 10,
        gens: int = 10,
    ):
        super().__init__(handler, oracle, seeds, iterations=iterations)
        self.collector = collector
        self.gens = gens

    @abstractmethod
    def interest(self, args: List[str], features: FeatureVector) -> bool:
        return False

    @abstractmethod
    def generate(self) -> List[str]:
        return list()

    @abstractmethod
    def oracle(self, args: List[str], features: FeatureVector) -> Label:
        return Label.NO_BUG

    def get_events(self, inputs: List[List[str]]):
        return self.collector.get_events(
            inputs,
            label=TestResult.PASSING,
        )

    def update_corpus(self, args: List[str], features: FeatureVector):
        pass

    def iteration(self):
        inputs = [self.generate() for _ in range(self.gens)]
        event_files = self.get_events(inputs)
        for args, event_file in zip(inputs, event_files):
            event_file.run_id = max(self.features.run_ids()) + 1
            self.handler.handle(event_file)
            features = self.features.get_vector_by_id(event_file.run_id)
            self.update_corpus(args, features)
            if self.interest(args, features):
                if self.oracle(args, features) == Label.BUG:
                    features.result = TestResult.FAILING
                else:
                    features.result = TestResult.PASSING
            else:
                self.features.remove(event_file.run_id)


class AFLTestGenFeedback(TestGenFeedback, ABC):
    def __init__(
        self,
        handler: Handler,
        oracle: Oracle,
        seeds: Dict[int, List[str]],
        collector: EventCollector,
        iterations: int = 10,
        gens: int = 10,
    ):
        super().__init__(
            handler, oracle, seeds, collector, iterations=iterations, gens=gens
        )
        self.coverage = set()
        self.corpus = [
            (self.seeds[run_id], self.features.get_vector_by_id(run_id))
            for run_id in self.seeds
        ]
        self.update_coverage()

    def update_coverage(self):
        self.coverage = set()
        for _, features in self.corpus:
            if features:
                self.coverage.add(features.tuple(self.all_features))

    def update_corpus(self, args: List[str], features: FeatureVector):
        feature_coverage = features.tuple(self.all_features)
        if feature_coverage not in self.features:
            self.corpus.append((args, features))

    def generate(self) -> List[str]:
        return list()
