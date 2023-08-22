import csv
import os
import shutil
from pathlib import Path
from unittest import TestCase

import tests4py.api as t4p
from sflkit.model import EventFile
from sflkit.runners.run import TestResult
from tests4py.constants import DEFAULT_WORK_DIR
from tests4py.projects import Project

from stato.events import instrument, get_t4p_events
from stato.features import Handler


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
        get_t4p_events(
            self.TEST_DIR,
            self.EVENTS_DIR,
            [
                ["2", "1", "3"],
                ["0", "-1", "1"],
                ["2", "4", "5"],
                ["63", "125", "64"],
                ["12", "-3", "-45"],
                ["2", "3", "1"],
                ["0", "1", "-1"],
                ["0", "1", "64"],
                ["4", "6", "5"],
                ["-1000", "0", "1000"],
            ],
        )
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

        all_features = list(handler.feature_builder.all_features)
        print(f"Found {len(all_features)} features")
        header = ["label"] + [feature.name for feature in all_features]

        with open("data.csv", "w") as fp:
            writer = csv.DictWriter(fp, fieldnames=header)
            writer.writeheader()
            for feature_vector in handler.feature_builder:
                num_dict = feature_vector.num_dict_vector(all_features)
                num_dict["label"] = (
                    1 if feature_vector.result == TestResult.FAILING else 0
                )
                writer.writerow(num_dict)
