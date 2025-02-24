import random
import string
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Callable

from sflkit.features.handler import EventHandler
from sflkit.features.vector import FeatureVector
from sflkit.model.model import Model
from sflkit.runners.run import TestResult

from bashiri.events import EventCollector
from bashiri.learning import Oracle, Label


def copy(self):
    new_handler = EventHandler()
    new_handler.builder = self.builder.copy()
    new_handler.model = Model(new_handler.builder)
    return new_handler


EventHandler.copy = copy


class RefinementLoop(ABC):
    def __init__(
        self,
        handler: EventHandler,
        oracle: Oracle,
        seeds: Dict[int, str],
        iterations: int = 10,
    ):
        self.handler = handler
        self.features = handler.builder
        self.learned_oracle = oracle
        self.seeds = dict()
        for s in seeds:
            self.seeds[s] = Seed(seeds[s])
        self.iterations = iterations
        self.all_features = self.features.all_features
        self.new_feature_vectors: List[FeatureVector] = list()

    @abstractmethod
    def iteration(self):
        pass

    def run(self):
        for _ in range(self.iterations):
            self.iteration()
            self.all_features = self.features.all_features
        if self.new_feature_vectors:
            self.learned_oracle.finalize(self.new_feature_vectors)


class TestGenRefinement(RefinementLoop, ABC):
    def __init__(
        self,
        handler: EventHandler,
        oracle: Oracle,
        seeds: Dict[int, str],
        collector: EventCollector,
        iterations: int = 10,
        gens: int = 10,
    ):
        super().__init__(handler, oracle, seeds, iterations=iterations)
        self.collector = collector
        self.gens = gens

    @abstractmethod
    def interest(self, args: str, features: FeatureVector) -> bool:
        return False

    @abstractmethod
    def generate(self) -> str:
        return ""

    @abstractmethod
    def oracle(self, args: str, features: FeatureVector) -> Label:
        return Label.NO_BUG

    def get_events(self, inputs: List[str]):
        return self.collector.get_events(
            inputs,
            label=TestResult.PASSING,
        )

    def update_corpus(self, args: str, features: FeatureVector):
        pass

    def iteration(self):
        inputs = [self.generate() for _ in range(self.gens)]
        event_files = self.get_events(inputs)
        for args, event_file in zip(inputs, event_files):
            event_file.run_id = max(self.features.run_ids()) + 1
            self.handler.handle(event_file)
            features = self.features.get_vector_by_id(event_file.run_id)
            if self.interest(args, features):
                if self.oracle(args, features) == Label.BUG:
                    features.result = TestResult.FAILING
                else:
                    features.result = TestResult.PASSING
                self.new_feature_vectors.append(features)
                self.seeds[event_file.run_id] = Seed(args)
            else:
                self.features.remove(event_file.run_id)
            self.update_corpus(args, features)


class MutationTestGenRefinement(TestGenRefinement, ABC):
    def __init__(
        self,
        handler: EventHandler,
        oracle: Oracle,
        seeds: Dict[int, List[str]],
        collector: EventCollector,
        iterations: int = 10,
        gens: int = 10,
        min_mutations: int = 1,
        max_mutations: int = 10,
    ):
        super().__init__(
            handler, oracle, seeds, collector, iterations=iterations, gens=gens
        )
        if min_mutations > max_mutations:
            raise ValueError(
                f"min_mutations needs to be smaller than max_mutations but was "
                f"{min_mutations} > {max_mutations}"
            )
        self.min_mutations = min_mutations
        self.max_mutations = max_mutations
        self.mutators: List[Callable[[str], str]] = self.get_mutators()

    @abstractmethod
    def select(self) -> str:
        pass

    def mutate(self, seed: str) -> str:
        s = self.prepare(seed)
        for _ in range(random.randint(self.min_mutations, self.max_mutations)):
            s = random.choice(self.mutators)(s)
        return Seed(s)

    def generate(self) -> str:
        return self.mutate(self.select())

    @abstractmethod
    def get_mutators(self) -> List[Callable[[str], str]]:
        pass

    @abstractmethod
    def prepare(self, seed):
        pass


class StringMutationTestGenRefinement(MutationTestGenRefinement, ABC):
    def __init__(
        self,
        handler: EventHandler,
        oracle: Oracle,
        seeds: Dict[int, List[str]],
        collector: EventCollector,
        iterations: int = 10,
        gens: int = 10,
        min_mutations: int = 1,
        max_mutations: int = 10,
        max_range: int = 10,
    ):
        super().__init__(
            handler,
            oracle,
            seeds,
            collector,
            iterations=iterations,
            gens=gens,
            min_mutations=min_mutations,
            max_mutations=max_mutations,
        )
        self.max_range = max_range
        self.numbers = list(
            map(
                str,
                [
                    -128,
                    -1,
                    0,
                    1,
                    16,
                    32,
                    64,
                    100,
                    127,
                    128,
                    255,
                    256,
                    512,
                    1000,
                    1024,
                    4096,
                    32767,
                    65535,
                ],
            )
        )

    def prepare(self, seed: str):
        return str(seed)

    @staticmethod
    def random_char() -> str:
        return random.choice(string.printable)

    def interesting_number(self) -> str:
        return random.choice(self.numbers)

    def delete_random_characters(self, s: str) -> str:
        if s == "":
            return s
        pos = random.randint(0, len(s) - 1)
        end = pos + min(len(s) - pos + 1, random.randint(1, self.max_range))
        return s[:pos] + s[end:]

    def insert_random_characters(self, s: str) -> str:
        pos = random.randint(0, len(s))
        return (
            s[:pos]
            + "".join(
                self.random_char() for _ in range(random.randint(1, self.max_range))
            )
            + s[pos + 1 :]
        )

    @staticmethod
    def flip_random_character(s: str) -> str:
        if s == "":
            return s
        pos = random.randint(0, len(s) - 1)
        c = s[pos]
        bit = 1 << random.randint(0, 6)
        new_c = chr(ord(c) ^ bit)
        return s[:pos] + new_c + s[pos + 1 :]

    def duplicate_random_characters(self, s: str) -> str:
        if s == "":
            return s
        pos = random.randint(0, len(s) - 1)
        end = pos + min(len(s) - pos + 1, random.randint(1, self.max_range))
        dst = random.randint(0, len(s) - 1)
        return s[:dst] + s[pos:end] + s[dst + 1 :]

    def replace_random_number(self, s: str) -> str:
        if s == "":
            return s
        current_number = -1
        numbers: List[Tuple[int, int]] = list()
        for i in range(len(s)):
            if s[i].isdigit():
                if current_number < 0:
                    current_number = i
            elif current_number >= 0:
                numbers.append((current_number, i))
                current_number = -1
        else:
            if current_number >= 0:
                numbers.append((current_number, len(s)))
        if numbers:
            start, end = random.choice(numbers)
            return s[:start] + self.interesting_number() + s[end:]
        else:
            return s

    def get_mutators(self) -> List[Callable[[str], str]]:
        return [
            self.delete_random_characters,
            self.insert_random_characters,
            self.flip_random_character,
            self.duplicate_random_characters,
            self.replace_random_number,
        ]


class Seed(str):
    def __new__(cls, args: str, energy: float = 1):
        obj = str.__new__(cls, args)
        obj.energy = energy
        return obj


class AFLRefinement(StringMutationTestGenRefinement, ABC):
    def __init__(
        self,
        handler: EventHandler,
        oracle: Oracle,
        seeds: Dict[int, List[str]],
        collector: EventCollector,
        iterations: int = 10,
        gens: int = 10,
        min_mutations: int = 1,
        max_mutations: int = 10,
    ):
        super().__init__(
            handler,
            oracle,
            seeds,
            collector,
            iterations=iterations,
            gens=gens,
            min_mutations=min_mutations,
            max_mutations=max_mutations,
        )
        self.coverage = set()
        self.corpus: List[Tuple[Seed, FeatureVector]] = [
            (self.seeds[run_id], self.features.get_vector_by_id(run_id))
            for run_id in self.seeds
        ]

    def assign_energy(self):
        for seed, _ in self.corpus:
            seed.energy = 1

    def normalized_energies(self):
        energies = list(map(lambda seed: seed[0].energy, self.corpus))
        sum_energy = sum(energies)
        if sum_energy <= 0:
            return 1
        return list(map(lambda energy: energy / sum_energy, energies))

    def select(self) -> str:
        self.assign_energy()
        norm_energy = self.normalized_energies()
        seed, _ = random.choices(self.corpus, weights=norm_energy)[0]
        return seed

    def update_coverage(self):
        self.coverage = set()
        for _, features in self.corpus:
            if features:
                self.coverage.add(features.tuple(self.all_features))

    def update_corpus(self, args: str, features: FeatureVector):
        feature_coverage = features.tuple(self.all_features)
        if feature_coverage not in self.features:
            self.corpus.append((args, features))


class AFLFastRefinement(AFLRefinement, ABC):
    def __init__(
        self,
        handler: EventHandler,
        oracle: Oracle,
        seeds: Dict[int, List[str]],
        collector: EventCollector,
        iterations: int = 10,
        gens: int = 10,
        min_mutations: int = 1,
        max_mutations: int = 10,
        exponent: float = 5,
    ):
        super().__init__(
            handler,
            oracle,
            seeds,
            collector,
            iterations=iterations,
            gens=gens,
            min_mutations=min_mutations,
            max_mutations=max_mutations,
        )
        self.exponent = exponent
        self.frequencies = dict()
        self.id_mapping = dict()
        for seed, features in self.corpus:
            self.update_frequencies(seed, features)

    def update_frequencies(self, seed: Seed, features: FeatureVector):
        t = features.tuple(self.all_features)
        self.id_mapping[seed] = t
        if t in self.frequencies:
            self.frequencies[t] += 1
        else:
            self.frequencies[t] = 1

    def assign_energy(self):
        for seed, features in self.corpus:
            seed.energy = 1 / (self.frequencies[self.id_mapping[seed]] ** self.exponent)

    def update_corpus(self, args: str, features: FeatureVector):
        super().update_corpus(args, features)
        self.update_frequencies(args, features)


class DifferenceInterestRefinement(AFLFastRefinement, ABC):
    def __init__(
        self,
        handler: EventHandler,
        oracle: Oracle,
        seeds: Dict[int, str],
        collector: EventCollector,
        iterations: int = 10,
        gens: int = 10,
        min_mutations: int = 1,
        max_mutations: int = 10,
        exponent: float = 5,
        threshold: float = 0.05,
    ):
        super().__init__(
            handler,
            oracle,
            seeds,
            collector,
            iterations=iterations,
            gens=gens,
            min_mutations=min_mutations,
            max_mutations=max_mutations,
            exponent=exponent,
        )
        self.threshold = threshold

    def interest(self, args: str, features: FeatureVector) -> bool:
        seeds = set(self.seeds.values())
        for seed, features_2 in self.corpus:
            if seed in seeds and (
                features.difference(features_2, self.all_features)
                / len(self.all_features)
                < self.threshold
            ):
                return False
        return True


class HumanOracleRefinement(DifferenceInterestRefinement, ABC):
    YES = ["y", "yes", "1"]
    NO = ["n", "no", "0"]

    def oracle(self, args: str, features: FeatureVector) -> Label:
        human_answer = None
        while human_answer not in self.YES + self.NO:
            print("Please answer the following question with yes[y,1] or no[n,0].")
            human_answer = (
                input(f"Is {args} a buggy input to the program under test: ")
                .lower()
                .strip()
            )
        if human_answer in self.YES:
            return Label.BUG
        else:
            return Label.NO_BUG
