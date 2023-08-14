import os
from os import PathLike
from typing import List, Dict, Optional

from sflkit import instrument_config, Config
from sflkit.runners import PytestRunner, InputRunner
from sflkitlib.events import EventType

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
