import json
import os
import time
import unittest
from pathlib import Path

import tests4py.api as t4p
from tests4py import sfl
from tests4py.projects import Project

from bashiri.events import Tests4PyEventCollector
from bashiri.features import Handler
from bashiri.refinement import Tests4PyEvaluationFeedbackLoop
from bashiri.learning import DecisionTreeOracle


class TestOBE(unittest.TestCase):
    TMP = Path("tmp_tests")
    TEST_DIR = TMP / "test_dir"
    EVENTS_DIR = TMP / "events"
    EVENTS_TRAINING = EVENTS_DIR / "train"
    EVENTS_EVALUATION = EVENTS_DIR / "eval"
    TRAINING = TMP / "train"
    EVALUATION = TMP / "eval"

    def test_middle_1(self):
        project: Project = t4p.middle_1
        project.buggy = True
        report = t4p.checkout_project(project)
        if report.raised:
            raise report.raised
        location = Path(report.location)
        report = sfl.sflkit_instrument(location, self.TEST_DIR)
        if report.raised:
            raise report.raised
        report = t4p.system_generate_project(
            self.TEST_DIR, self.EVALUATION, n=200, p=100
        )
        eval_inputs = []
        if report.raised:
            raise report.raised
        for file in os.listdir(self.TRAINING):
            with open(self.TRAINING / file, "r") as fp:
                eval_inputs.append(fp.read())
        eval_collector = Tests4PyEventCollector(self.TEST_DIR)
        eval_events = eval_collector.get_events(eval_inputs)
        eval_handler = Handler()
        eval_handler.handle_files(eval_events)

        results = dict()

        for t, f in ((5, 1), (10, 1), (10, 2), (30, 3)):
            result = dict()

            report = t4p.system_generate_project(self.TEST_DIR, self.TRAINING, n=t, p=f)
            if report.raised:
                raise report.raised

            inputs = []
            for file in os.listdir(self.TRAINING):
                with open(self.TRAINING / file, "r") as fp:
                    inputs.append(fp.read())
            obe_time = time.time()
            collector = Tests4PyEventCollector(self.TEST_DIR)
            events = collector.get_events(inputs)
            handler = Handler()
            handler.handle_files(events)
            all_features = handler.feature_builder.get_all_features()
            path = Path("dt")
            if path.exists():
                os.remove(path)
            oracle = DecisionTreeOracle(path=path)
            oracle.fit(
                all_features,
                handler.feature_builder.get_vectors(),
            )
            obe_time = time.time() - obe_time
            report_eval = oracle.evaluate(
                eval_handler.feature_builder.get_vectors(),
                output_dict=True,
            )
            result["eval"] = report_eval
            result["time"] = obe_time
            print(report_eval)

            oracle_10 = oracle
            oracle_100 = DecisionTreeOracle(path=path)
            oracle_100.fit(
                all_features,
                handler.feature_builder.get_vectors(),
            )

            feedback_time_10 = time.time()
            feedbackloop = Tests4PyEvaluationFeedbackLoop(
                handler,
                oracle_10,
                {i: args for i, args in enumerate(inputs)},
                collector,
            )
            feedbackloop.run()
            feedback_time_10 = time.time() - feedback_time_10
            report_10 = oracle.evaluate(
                eval_handler.feature_builder.get_vectors(), output_dict=True
            )
            result["eval_fb_10"] = report_10
            result["time_fb_10"] = feedback_time_10
            print(report_10)

            feedback_time_100 = time.time()
            feedbackloop = Tests4PyEvaluationFeedbackLoop(
                handler,
                oracle_100,
                {i: args for i, args in enumerate(inputs)},
                collector,
                gens=100,
            )
            feedbackloop.run()
            feedback_time_100 = time.time() - feedback_time_100
            report_100 = oracle.evaluate(
                eval_handler.feature_builder.get_vectors(), output_dict=True
            )
            result["eval_fb_100"] = report_100
            result["time_fb_100"] = feedback_time_100
            print(report_100)

            results[f"tests_{t}_failing_{f}"] = result

        with open(f"{project.get_identifier()}.json", "w") as fp:
            json.dump(result, fp)
