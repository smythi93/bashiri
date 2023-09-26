import argparse
import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
import shlex
from typing import Optional, Sequence, Any, Dict, List

import sflkit.logger
import tests4py.logger
from tqdm import tqdm

import tests4py.api as t4p
from sflkit.runners.run import TestResult
from tests4py import sfl
from tests4py.api import run_project
from tests4py.projects import Project

from bashiri.events import EventCollector
from bashiri.features import Handler, FeatureVector
from bashiri.refinement import (
    DifferenceInterestRefinement,
)
from bashiri.learning import DecisionTreeOracle, Oracle, Label

TMP = Path("../tmp_tests")
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

    for t, f in tqdm(((5, 1), (10, 1), (10, 2), (30, 3))):
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
        path = Path("../dt")
        if path.exists():
            os.remove(path)
        oracle = DecisionTreeOracle(path=path)
        oracle.fit(
            all_features,
            handler.feature_builder.get_vectors(),
        )
        obe_time = time.time() - obe_time
        report_eval, confusion = oracle.evaluate(
            eval_handler.feature_builder.get_vectors(),
            output_dict=True,
        )
        result["eval"] = report_eval
        result["time"] = obe_time
        result["confusion"] = confusion

        oracle_10 = DecisionTreeOracle(path=path)
        oracle_100 = DecisionTreeOracle(path=path)
        oracle_10.fit(
            all_features,
            handler.feature_builder.get_vectors(),
        )
        oracle_100.fit(
            all_features,
            handler.feature_builder.get_vectors(),
        )

        refinement_time_10 = time.time()
        refinement_loop = Tests4PyEvaluationRefinement(
            handler,
            oracle_10,
            {i: args for i, args in enumerate(inputs)},
            collector,
        )
        refinement_loop.run()
        refinement_time_10 = time.time() - refinement_time_10
        report_10, confusion_10 = oracle_10.evaluate(
            eval_handler.feature_builder.get_vectors(), output_dict=True
        )
        result["eval_fb_10"] = report_10
        result["time_fb_10"] = refinement_time_10
        result["confusion_fb_10"] = refinement_time_10
        result["new_fb_10"] = len(refinement_loop.new_feature_vectors)
        result["failing_fb_10"] = len(
            list(
                filter(
                    lambda vector: vector.result == TestResult.FAILING,
                    refinement_loop.new_feature_vectors,
                )
            )
        )
        report_10_new, confusion_10_new = oracle.evaluate(
            refinement_loop.new_feature_vectors, output_dict=True
        )
        result["eval_fb_10_new"] = report_10_new
        result["confusion_fb_10_new"] = confusion_10_new

        refinement_time_100 = time.time()
        refinement_loop = Tests4PyEvaluationRefinement(
            handler,
            oracle_100,
            {i: arguments for i, arguments in enumerate(inputs)},
            collector,
            gens=100,
        )
        refinement_loop.run()
        refinement_time_100 = time.time() - refinement_time_100
        report_100, confusion_100 = oracle.evaluate(
            eval_handler.feature_builder.get_vectors(), output_dict=True
        )
        result["eval_fb_100"] = report_100
        result["time_fb_100"] = refinement_time_100
        result["confusion_fb_100"] = confusion_100
        result["new_fb_100"] = len(refinement_loop.new_feature_vectors)
        result["failing_fb_100"] = len(
            list(
                filter(
                    lambda vector: vector.result == TestResult.FAILING,
                    refinement_loop.new_feature_vectors,
                )
            )
        )
        report_100_new, confusion_100_new = oracle.evaluate(
            refinement_loop.new_feature_vectors, output_dict=True
        )
        result["eval_fb_100_new"] = report_100_new
        result["confusion_fb_100_new"] = confusion_100_new

        results[f"tests_{t}_failing_{f}"] = result

    with open(f"{project.get_identifier()}.json", "w") as fp:
        json.dump(results, fp, indent=2)


class Tests4PyEventCollector(EventCollector):
    @staticmethod
    def convert(test_result: TestResult) -> TestResult:
        if test_result.name == "PASSING":
            return TestResult.PASSING
        elif test_result.name == "FAILING":
            return TestResult.FAILING
        else:
            return TestResult.UNDEFINED

    def collect(
        self,
        output: os.PathLike,
        tests: Optional[Sequence[Any] | str] = None,
        label: Optional[TestResult] = None,
    ):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        for test_result in TestResult:
            (output / test_result.get_dir()).mkdir(parents=True, exist_ok=True)
        for test in tests:
            test_input = test
            if isinstance(test, str):
                try:
                    test_input = shlex.split(test)
                except ValueError:
                    pass
            if label is None:
                report = run_project(self.work_dir, test_input, invoke_oracle=True)
                if report.raised:
                    test_result = TestResult.UNDEFINED
                else:
                    test_result = self.convert(report.test_result)
            else:
                report = run_project(self.work_dir, test_input)
                if report.raised:
                    raise report.raised
                test_result = label
            self.runs[test] = test_result
            if os.path.exists(self.work_dir / "EVENTS_PATH"):
                shutil.move(
                    self.work_dir / "EVENTS_PATH",
                    output
                    / test_result.get_dir()
                    / hashlib.md5(" ".join(test).encode("utf8")).hexdigest(),
                )


class Tests4PyEvaluationRefinement(DifferenceInterestRefinement):
    def __init__(
        self,
        handler: Handler,
        oracle: Oracle,
        seeds: Dict[int, str],
        collector: EventCollector,
        iterations: int = 10,
        gens: int = 10,
        min_mutations: int = 1,
        max_mutations: int = 10,
        exponent: float = 5,
        threshold: float = 0.05,
    ):
        super().__init__(
            handler,
            oracle,
            seeds,
            collector,
            iterations=iterations,
            gens=gens,
            min_mutations=min_mutations,
            max_mutations=max_mutations,
            exponent=exponent,
            threshold=threshold,
        )
        assert isinstance(collector, Tests4PyEventCollector)

    def get_events(self, inputs: List[str]):
        return self.collector.get_events(
            inputs,
        )

    def oracle(self, args: str, features: FeatureVector) -> Label:
        if args in self.collector.runs:
            return (
                Label.BUG
                if self.collector.runs[args] == TestResult.FAILING
                else Label.NO_BUG
            )
        else:
            logging.warning(f"cannot find {args} in run")
        return Label.NO_BUG


def main(project_name: str, bug_id: int):
    project = t4p.get_projects(project_name, bug_id)[0]
    assert project.systemtests
    evaluate_project(project)


if __name__ == "__main__":
    tests4py.logger.LOGGER.disabled = True
    sflkit.logger.LOGGER.disabled = True
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
    arguments = arg_parser.parse_args()
    main(arguments.project_name, arguments.bug_id)
