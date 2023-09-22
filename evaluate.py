import argparse
import json
import os
import time
from pathlib import Path

import tests4py.api as t4p
from sflkit.runners.run import TestResult
from tests4py import sfl
from tests4py.projects import Project

from stato.events import Tests4PyEventCollector
from stato.features import Handler
from stato.feedback import Tests4PyEvaluationFeedbackLoop
from stato.learning import DecisionTree

TMP = Path("tmp_tests")
TEST_DIR = TMP / "test_dir"
EVENTS_DIR = TMP / "events"
EVENTS_TRAINING = EVENTS_DIR / "train"
EVENTS_EVALUATION = EVENTS_DIR / "eval"
TRAINING = TMP / "train"
EVALUATION = TMP / "eval"


def evaluate_project(project: Project):
    project.buggy = True
    report = t4p.checkout_project(project)
    if report.raised:
        raise report.raised
    location = Path(report.location)
    report = sfl.sflkit_instrument(location, TEST_DIR)
    if report.raised:
        raise report.raised
    report = t4p.system_generate_project(TEST_DIR, EVALUATION, n=200, p=100)
    eval_inputs = []
    if report.raised:
        raise report.raised
    for file in os.listdir(EVALUATION):
        with open(EVALUATION / file, "r") as fp:
            eval_inputs.append(fp.read())
    eval_collector = Tests4PyEventCollector(TEST_DIR)
    eval_events = eval_collector.get_events(eval_inputs)
    eval_handler = Handler()
    eval_handler.handle_files(eval_events)

    results = dict()

    for t, f in ((5, 1), (10, 1), (10, 2), (30, 3)):
        result = dict()

        report = t4p.system_generate_project(TEST_DIR, TRAINING, n=t, p=f)
        if report.raised:
            raise report.raised

        inputs = []
        for file in os.listdir(TRAINING):
            with open(TRAINING / file, "r") as fp:
                inputs.append(fp.read())
        obe_time = time.time()
        collector = Tests4PyEventCollector(TEST_DIR)
        events = collector.get_events(inputs)
        handler = Handler()
        handler.handle_files(events)
        all_features = handler.feature_builder.get_all_features()
        path = Path("dt")
        if path.exists():
            os.remove(path)
        oracle = DecisionTree(path=path)
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

        oracle_10 = DecisionTree(path=path)
        oracle_100 = DecisionTree(path=path)
        oracle_10.fit(
            all_features,
            handler.feature_builder.get_vectors(),
        )
        oracle_100.fit(
            all_features,
            handler.feature_builder.get_vectors(),
        )

        feedback_time_10 = time.time()
        feedbackloop = Tests4PyEvaluationFeedbackLoop(
            handler,
            oracle_10,
            {i: arguments for i, arguments in enumerate(inputs)},
            collector,
        )
        feedbackloop.run()
        feedback_time_10 = time.time() - feedback_time_10
        report_10 = oracle.evaluate(
            eval_handler.feature_builder.get_vectors(), output_dict=True
        )
        result["eval_fb_10"] = report_10
        result["time_fb_10"] = feedback_time_10
        result["new_fb_10"] = len(feedbackloop.new_feature_vectors)
        result["failing_fb_10"] = len(
            list(
                filter(
                    lambda vector: vector.result == TestResult.FAILING,
                    feedbackloop.new_feature_vectors,
                )
            )
        )
        print(report_10)

        feedback_time_100 = time.time()
        feedbackloop = Tests4PyEvaluationFeedbackLoop(
            handler,
            oracle_100,
            {i: arguments for i, arguments in enumerate(inputs)},
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
        result["new_fb_100"] = len(feedbackloop.new_feature_vectors)
        result["failing_fb_100"] = len(
            list(
                filter(
                    lambda vector: vector.result == TestResult.FAILING,
                    feedbackloop.new_feature_vectors,
                )
            )
        )
        print(report_100)

        results[f"tests_{t}_failing_{f}"] = result

    with open(f"{project.get_identifier()}.json", "w") as fp:
        json.dump(results, fp)


def main(project_name: str, bug_id: int):
    project = t4p.get_projects(project_name, bug_id)[0]
    assert project.systemtests
    evaluate_project(project)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-p",
        dest="project_name",
        required=True,
        help="The name of the project to evaluate",
    )
    arg_parser.add_argument(
        "-i",
        dest="bug_id",
        type=int,
        required=True,
        help="The bug id of the project to evaluate",
    )
    args = arg_parser.parse_args()
    main(args.project_name, args.bug_id)
