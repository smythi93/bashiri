import os
import shutil
from abc import abstractmethod, ABC
from os import PathLike
from pathlib import Path
from typing import List, Dict, Optional, Sequence, Any, Tuple

from sflkit import instrument_config, Config
from sflkit.config import hash_identifier
from sflkit.events.event_file import EventFile
from sflkit.events.mapping import EventMapping
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
    src: PathLike,
    dst: PathLike,
    mapping_path: PathLike,
    excludes: List[str] = None,
    events: List[str] = None,
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
            mapping_path=str(mapping_path),
        ),
    )


class EventCollector(ABC):
    def __init__(
        self,
        work_dir: PathLike,
        src: PathLike,
        mapping: Optional[EventMapping | PathLike] = None,
    ):
        self.work_dir = Path(work_dir)
        self.src = Path(src)
        self.runs: Dict[str, TestResult] = dict()
        if isinstance(mapping, EventMapping):
            self.mapping = mapping
        else:
            self.mapping = EventMapping.load_from_file(
                Path(mapping or EventMapping.get_path(self.identifier())),
            )

    def identifier(self):
        return hash_identifier(self.work_dir)

    @abstractmethod
    def collect(
        self,
        output: PathLike,
        tests: Optional[Sequence[Any]] = None,
        label: Optional[TestResult] = None,
    ):
        pass

    def get_event_files(
        self, events: PathLike
    ) -> Tuple[List[EventFile], List[EventFile]]:
        events = Path(events)
        failing = [
            EventFile(
                events / "failing" / path,
                run_id,
                self.mapping,
                failing=True,
            )
            for run_id, path in enumerate(os.listdir(events / "failing"), start=0)
        ]
        passing = [
            EventFile(events / "passing" / path, run_id, self.mapping, failing=False)
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
    def __init__(
        self,
        work_dir: PathLike,
        src: PathLike,
        environ: Optional[Dict[str, str]] = None,
        mapping: Optional[EventMapping | PathLike] = None,
    ):
        super().__init__(work_dir, src, mapping=mapping)
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
        src: PathLike,
        access: PathLike,
        environ: Optional[Dict[str, str]] = None,
        mapping: Optional[EventMapping | PathLike] = None,
    ):
        super().__init__(work_dir, src, mapping=mapping)
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
