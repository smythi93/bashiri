import hashlib
import os
import shutil
from os import PathLike
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Sequence

from sflkit import instrument_config, Config
from sflkit.model import EventFile
from sflkit.runners import PytestRunner, InputRunner
from sflkit.runners.run import TestResult
from sflkitlib.events import EventType

from tests4py.api import run_project

EVENTS = list(map(lambda e: e.name, list(EventType)))

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


def instrument(src: PathLike, dst: PathLike, excludes: List[str] = None):
    if excludes is None:
        excludes = DEFAULT_EXCLUDES
    instrument_config(
        Config.create(
            path=str(src),
            language="Python",
            events=",".join(EVENTS),
            working=str(dst),
            exclude=",".join(excludes),
        ),
    )


def get_events_unittests(
    work_dir: PathLike, output: PathLike, environ: Optional[Dict[str, str]] = None
):
    if environ is None:
        environ = os.environ
    PytestRunner().run(directory=work_dir, output=output, environ=environ)


def get_events_systemtests(
    work_dir: PathLike,
    output: PathLike,
    access: PathLike,
    passing: List[str | List[str]],
    failing: List[str | List[str]],
    environ: Optional[Dict[str, str]] = None,
):
    if environ is None:
        environ = os.environ
    runner = InputRunner(access, passing, failing)
    runner.run(directory=work_dir, output=output, environ=environ)


def convert(test_result: TestResult) -> TestResult:
    if test_result.name == "PASSING":
        return TestResult.PASSING
    elif test_result.name == "FAILING":
        return TestResult.FAILING
    else:
        return TestResult.UNDEFINED


def get_t4p_events(
    directory: Path,
    output: Path,
    tests: Sequence[Sequence[str] | PathLike],
    label: Optional[TestResult] = None,
):
    output.mkdir(parents=True, exist_ok=True)
    for test_result in TestResult:
        (output / test_result.get_dir()).mkdir(parents=True, exist_ok=True)
    for test in tests:
        if label is None:
            report = run_project(directory, test, invoke_oracle=True)
            if report.raised:
                raise report.raised
            test_result = convert(report.test_result)
        else:
            report = run_project(directory, test)
            if report.raised:
                raise report.raised
            test_result = label
        if os.path.exists(directory / "EVENTS_PATH"):
            shutil.move(
                directory / "EVENTS_PATH",
                output
                / test_result.get_dir()
                / hashlib.md5(" ".join(test).encode("utf8")).hexdigest(),
            )


def get_event_files(events: Path) -> Tuple[List[EventFile], List[EventFile]]:
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
