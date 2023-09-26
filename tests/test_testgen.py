import unittest

from bashiri.features import FeatureVector, EventHandler
from bashiri.refinement import StringMutationTestGenRefinement
from bashiri.learning import Label


class TestTestGen(unittest.TestCase):
    def test_string(self):
        class Fuzzer(StringMutationTestGenRefinement):
            def select(self) -> str:
                return ""

            def interest(self, args: str, features: FeatureVector) -> bool:
                return False

            def oracle(self, args: str, features: FeatureVector) -> Label:
                return Label.NO_BUG

        fuzzer = Fuzzer(
            EventHandler(), None, dict(), None, max_mutations=1, max_range=1
        )

        for _ in range(30):
            print(fuzzer.mutate("2 1 3").encode("utf-8"))
