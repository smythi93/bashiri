import os
import shutil
from pathlib import Path
from unittest import TestCase

import tests4py.api as t4p
from sflkit.model import EventFile
from tests4py.constants import DEFAULT_WORK_DIR
from tests4py.projects import Project

from statio.events import instrument, get_events_unittests, get_events_systemtests
from statio.features import Handler


class TestFeatures(TestCase):
    TMP = Path("tmp_tests")
    TEST_DIR = TMP / "test_dir"
    EVENTS_DIR = TMP / "events"

    def tearDown(self) -> None:
        shutil.rmtree(DEFAULT_WORK_DIR, ignore_errors=True)
        shutil.rmtree(self.TMP, ignore_errors=True)

    def test_middle(self):
        project: Project = t4p.middle_1
        project.buggy = True
        report = t4p.checkout_project(project)
        if report.raised:
            raise report.raised
        location = Path(report.location)
        instrument(
            location,
            self.TEST_DIR,
        )
        report = t4p.compile_project(self.TEST_DIR, sfl=True)
        if report.raised:
            raise report.raised
        get_events_unittests(self.TEST_DIR, self.EVENTS_DIR, environ=report.env)
        failing = [
            EventFile(self.EVENTS_DIR / "failing" / path, run_id, failing=True)
            for run_id, path in enumerate(
                os.listdir(self.EVENTS_DIR / "failing"), start=0
            )
        ]
        passing = [
            EventFile(self.EVENTS_DIR / "passing" / path, run_id)
            for run_id, path in enumerate(
                os.listdir(self.EVENTS_DIR / "passing"), start=len(failing)
            )
        ]
        handler = Handler()
        handler.handle_files(failing)
        handler.handle_files(passing)

        for feature_vector in handler.feature_builder:
            print(feature_vector)
