import csv
import os
import shutil
from pathlib import Path
from unittest import TestCase

import tests4py.api as t4p
from sflkit.runners.run import TestResult
from tests4py import sfl
from tests4py.constants import DEFAULT_WORK_DIR
from tests4py.projects import Project

from bashiri.events import Tests4PyEventCollector, OUTPUT
from bashiri.features import EventHandler


class TestFeatures(TestCase):
    TMP = Path("tmp_tests")
    TEST_DIR = TMP / "test_dir"
    EVENTS_DIR = TMP / "events"
    EVENTS_TRAINING = EVENTS_DIR / "train"
    EVENTS_EVALUATION = EVENTS_DIR / "eval"
    TRAINING = TMP / "train"
    EVALUATION = TMP / "eval"

    def tearDown(self) -> None:
        shutil.rmtree(DEFAULT_WORK_DIR, ignore_errors=True)
        shutil.rmtree(self.TMP, ignore_errors=True)
        shutil.rmtree(OUTPUT, ignore_errors=True)

    def _test_project(self, project: Project):
        project.buggy = True
        report = t4p.checkout_project(project)
        if report.raised:
            raise report.raised
        location = Path(report.location)
        report = sfl.sflkit_instrument(location, self.TEST_DIR)
        if report.raised:
            raise report.raised
        report = t4p.system_generate_project(self.TEST_DIR, self.TRAINING, n=10, p=2)
        if report.raised:
            raise report.raised
        report = t4p.system_generate_project(
            self.TEST_DIR, self.EVALUATION, n=100, p=20
        )
        if report.raised:
            raise report.raised
        all_features = None
        for label, base in (
            (f"train_{project.get_identifier()}", self.TRAINING),
            (f"eval_{project.get_identifier()}", self.EVALUATION),
        ):
            inputs = []
            for file in os.listdir(base):
                with open(base / file, "r") as fp:
                    inputs.append(fp.read().split("\n"))
            collector = Tests4PyEventCollector(self.TEST_DIR)
            events = collector.get_events(inputs)
            handler = EventHandler()
            handler.handle_files(events)
            if all_features is None:
                all_features = list(handler.feature_builder.get_all_features())
                print(f"Found {len(all_features)} features")
            header = ["label"] + [feature.name for feature in all_features]

            with open(f"{label}.csv", "w") as fp:
                writer = csv.DictWriter(fp, fieldnames=header)
                writer.writeheader()
                for feature_vector in handler.feature_builder:
                    num_dict = feature_vector.num_dict_vector(all_features)
                    num_dict["label"] = (
                        1 if feature_vector.result == TestResult.FAILING else 0
                    )
                    writer.writerow(num_dict)

    def test_middle_1(self):
        self._test_project(t4p.middle_1)

    def test_pysnooper_2(self):
        self._test_project(t4p.pysnooper_2)

    def test_pysnooper_3(self):
        self._test_project(t4p.pysnooper_3)

    def test_cookiecutter_2(self):
        self._test_project(t4p.cookiecutter_2)

    def test_cookiecutter_3(self):
        self._test_project(t4p.cookiecutter_3)

    def test_fastapi_1(self):
        self._test_project(t4p.fastapi_1)

    def test_fastapi_2(self):
        self._test_project(t4p.fastapi_2)
