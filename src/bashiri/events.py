import os
import shutil
from abc import abstractmethod, ABC
from os import PathLike
from pathlib import Path
from typing import List, Dict, Optional, Sequence, Any

from sflkit import instrument_config, Config
from sflkit.model import EventFile
from sflkit.runners import PytestRunner, InputRunner
from sflkit.runners.run import TestResult
from sflkitlib.events import EventType

EVENTS = list(map(lambda e: e.name, list(EventType)))
OUTPUT = Path("tmp_events")

DEFAULT_EXCLUDES = [
    "test",
    "tests",
    "setup.py",
    "env",
    "build",
    "bin",
    "docs",
    "examples",
    "hacking",
    ".git",
    ".github",
    "extras",
    "profiling",
    "plugin",
    "gallery",
    "blib2to3",
    "docker",
    "contrib",
    "changelogs",
    "licenses",
    "packaging",
]


def instrument(
    src: PathLike, dst: PathLike, excludes: List[str] = None, events: List[str] = None
):
    if excludes is None:
        excludes = DEFAULT_EXCLUDES
    instrument_config(
        Config.create(
            path=str(src),
            language="Python",
            events=",".join(events or EVENTS),
            working=str(dst),
            exclude=",".join(excludes),
        ),
    )


class EventCollector(ABC):
    def __init__(self, work_dir: PathLike):
        self.work_dir = Path(work_dir)
        self.runs: Dict[str, TestResult] = dict()

    @abstractmethod
    def collect(
        self,
        output: PathLike,
        tests: Optional[Sequence[Any]] = None,
        label: Optional[TestResult] = None,
    ):
        pass

    @staticmethod
    def get_event_files(events: PathLike) -> List[EventFile]:
        events = Path(events)
        failing = [
            EventFile(events / "failing" / path, run_id, failing=True)
            for run_id, path in enumerate(os.listdir(events / "failing"), start=0)
        ]
        passing = [
            EventFile(events / "passing" / path, run_id)
            for run_id, path in enumerate(
                os.listdir(events / "passing"),
                start=len(failing),
            )
        ]
        return failing, passing

    def get_events(
        self,
        tests: Any = None,
        label: Optional[TestResult] = None,
    ) -> List[EventFile]:
        shutil.rmtree(OUTPUT, ignore_errors=True)
        self.collect(OUTPUT, tests=tests, label=label)
        failing, passing = self.get_event_files(OUTPUT)
        return failing + passing


class UnittestEventCollector(EventCollector):
    def __init__(self, work_dir: PathLike, environ: Optional[Dict[str, str]] = None):
        super().__init__(work_dir)
        self.environ = environ

    def collect(
        self,
        output: PathLike,
        tests: Optional[Sequence[Any]] = None,
        label: Optional[TestResult] = None,
    ):
        if self.environ is None:
            self.environ = os.environ
        PytestRunner().run(directory=self.work_dir, output=output, environ=self.environ)


class SystemtestEventCollector(EventCollector):
    def __init__(
        self,
        work_dir: PathLike,
        access: PathLike,
        environ: Optional[Dict[str, str]] = None,
    ):
        super().__init__(work_dir)
        self.access = access
        self.environ = environ

    def collect(
        self,
        output: PathLike,
        tests: Any = None,
        label: Optional[TestResult] = None,
    ):
        if label is None:
            passing, failing = tests
        else:
            passing, failing = tests, list()
        if self.environ is None:
            self.environ = os.environ
        runner = InputRunner(self.access, passing, failing)
        runner.run(directory=self.work_dir, output=output, environ=self.environ)
