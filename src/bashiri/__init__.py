import enum
import os
import time
import warnings
from typing import Sequence, Any, Optional, Tuple

from sflkit.features.handler import EventHandler
from sflkit.runners.run import TestResult

from bashiri.events import (
    SystemtestEventCollector,
    UnittestEventCollector,
    EventCollector,
)
from bashiri.learning import Oracle, DecisionTreeOracle, Label
from bashiri.reduce import FeatureSelection, DefaultSelection


class Mode(enum.Enum):
    SYSTEM_TEST = "system_test"
    UNIT_TEST = "unit_test"
    CUSTOM = "custom"


class Bashiri(DecisionTreeOracle):
    def __init__(
        self,
        src: Optional[os.PathLike] = None,
        tests: Optional[Tuple[Sequence[Any], Sequence[Any]] | Sequence[Any]] = None,
        mode: Mode = Mode.SYSTEM_TEST,
        collector: EventCollector = None,
        access: Optional[os.PathLike] = None,
        path: Optional[os.PathLike] = None,
        reducer: FeatureSelection = DefaultSelection(),
        random_state: int = 42,
        criterion: str = "log_loss",
        work_dir: os.PathLike = "tmp",
        environ: dict[str, str] = None,
        mapping: Optional[os.PathLike] = None,
    ):
        super().__init__(
            path=path, reducer=reducer, random_state=random_state, criterion=criterion
        )
        if mode == Mode.SYSTEM_TEST:
            if access is None:
                raise ValueError("Access path must be provided for system tests.")
            if tests is None or not isinstance(tests, tuple) or len(tests) != 2:
                raise ValueError(
                    "Tests must be a tuple of (passing, failing) sequences."
                )
            if src is None:
                raise ValueError("Source path must be provided for system tests.")
            self.collector = SystemtestEventCollector(
                work_dir=work_dir,
                src=src,
                access=access,
                environ=environ,
                mapping=mapping,
            )
        elif mode == Mode.UNIT_TEST:
            if src is None:
                raise ValueError("Source path must be provided for unit tests.")
            self.collector = UnittestEventCollector(
                work_dir=work_dir,
                src=src,
                environ=environ,
                mapping=mapping,
            )
        elif mode == Mode.CUSTOM:
            if collector is None:
                raise ValueError("Collector must be provided for custom mode.")
            self.collector = collector
            if not isinstance(self.collector, EventCollector):
                raise TypeError("Collector must be an instance of EventCollector.")
        else:
            raise ValueError(f"Unknown mode: {mode}")
        self.tests = tests
        self.events = []
        self.handler = EventHandler()
        self.learned = False

    def learn(self):
        """
        Collects events and learns from them.
        """
        if self.learned:
            warnings.warn(
                "Bashiri has already learned from the events. "
                "Re-learning will overwrite the existing model.",
                UserWarning,
            )
        self.events = self.collector.get_events(self.tests)
        time.sleep(1)
        self.handler.handle_files(self.events)
        self.all_features = self.handler.builder.get_all_features()
        self.fit(
            self.all_features,
            self.handler,
        )
        self.learned = True

    def predict(self, test) -> Label:
        """
        Predicts the outcome of the given test using the learned model.
        """
        if not self.learned:
            raise RuntimeError("Bashiri has not learned from the events yet.")
        new_events = self.collector.get_events([test], label=TestResult.UNDEFINED)
        handler = EventHandler()
        handler.handle_files(new_events)
        x, _ = self.prepare_data(handler)
        return self.classify(x)
