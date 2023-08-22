import csv
import os
import shutil
from pathlib import Path
from unittest import TestCase

import tests4py.api as t4p
from sflkit.runners.run import TestResult
from tests4py.constants import DEFAULT_WORK_DIR
from tests4py.projects import Project

from stato.events import instrument, get_t4p_events, get_event_files
from stato.features import Handler


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
        report = t4p.system_generate_project(self.TEST_DIR, self.TRAINING, n=10, p=2)
        if report.raised:
            raise report.raised
        report = t4p.system_generate_project(
            self.TEST_DIR, self.EVALUATION, n=100, p=20
        )
        if report.raised:
            raise report.raised
        all_features = None
        for label, base, events in (
            ("train", self.TRAINING, self.EVENTS_TRAINING),
            ("eval", self.EVALUATION, self.EVENTS_EVALUATION),
        ):
            inputs = []
            for file in os.listdir(base):
                with open(base / file, "r") as fp:
                    inputs.append(fp.read().split("\n"))
            get_t4p_events(
                self.TEST_DIR,
                events,
                inputs,
            )
            failing, passing = get_event_files(events)
            handler = Handler()
            handler.handle_files(failing)
            handler.handle_files(passing)
            if all_features is None:
                all_features = list(handler.feature_builder.all_features)
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
