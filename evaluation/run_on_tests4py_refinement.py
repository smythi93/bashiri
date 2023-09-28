import argparse
import hashlib
import json
import logging
import os
import random
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
from bashiri.features import EventHandler, FeatureVector
from bashiri.refinement import (
    DifferenceInterestRefinement,
)
from bashiri.learning import DecisionTreeOracle, Oracle, Label

TMP = Path("tmp_tests")
TEST_DIR = TMP / "test_dir"
EVENTS_DIR = TMP / "events"
EVENTS_TRAINING = EVENTS_DIR / "train"
EVENTS_EVALUATION = EVENTS_DIR / "eval"
TRAINING = TMP / "train"
EVALUATION = TMP / "eval"

N = 200
F = 100
RLS = [100]
SEED = 42

RESULTS_PATH = Path("results")


def evaluate_project(project: Project):
    project.buggy = True
    logging.info(f"Checking out subject {project}")
    report = t4p.checkout_project(project)
    if report.raised:
        raise report.raised
    location = Path(report.location)
    logging.info(f"Instrumenting and compiling subject {project}")
    report = sfl.sflkit_instrument(location, TEST_DIR)
    if report.raised:
        raise report.raised
    logging.info(f"Generating {N} tests ({F} failing) for evaluation")
    report = t4p.system_generate_project(TEST_DIR, EVALUATION, n=N, p=F)
    eval_inputs = []
    if report.raised:
        raise report.raised
    for file in os.listdir(EVALUATION):
        with open(EVALUATION / file, "r") as fp:
            eval_inputs.append(fp.read())
    # do not split input with shlex for cookiecutter
    eval_collector = Tests4PyEventCollector(
        TEST_DIR, progress=True, split=project.project_name != "cookiecutter"
    )
    logging.info(f"Collecting events")
    eval_events = eval_collector.get_events(eval_inputs)
    eval_handler = EventHandler()
    logging.info(f"Constructing features")
    for e in tqdm(eval_events):
        eval_handler.handle(e)
    assert len(eval_handler.feature_builder.feature_vectors) == len(eval_events)

    results = dict()

    logging.info(f"Running configurations")
    for i in range(3):
        logging.info(f"Starting evaluation of {i}")
        result = dict()

        report = t4p.system_generate_project(TEST_DIR, TRAINING, n=2, p=1)
        if report.raised:
            raise report.raised

        inputs = []
        for file in os.listdir(TRAINING):
            with open(TRAINING / file, "r") as fp:
                inputs.append(fp.read())
        logging.info(f"Training and evaluating initial oracle")
        obe_time = time.time()
        # do not split input with shlex for cookiecutter
        collector = Tests4PyEventCollector(
            TEST_DIR, progress=True, split=project.project_name != "cookiecutter"
        )
        events = collector.get_events(inputs)
        handler = EventHandler()
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
        result["refinement"] = dict()

        for gens in RLS:
            logging.info(
                f"Refining and evaluating initial oracle with {gens} generations"
            )
            refinement_oracle = DecisionTreeOracle()
            refinement_oracle.fit(all_features, handler.feature_builder.get_vectors())
            refinement_time = time.time()
            refinement_loop = Tests4PyEvaluationRefinement(
                handler.copy(),
                refinement_oracle,
                {i: args for i, args in enumerate(inputs)},
                # do not split input with shlex for cookiecutter
                Tests4PyEventCollector(
                    TEST_DIR,
                    progress=False,
                    split=project.project_name != "cookiecutter",
                ),
                gens=gens,
            )
            refinement_loop.run()
            refinement_time = time.time() - refinement_time
            refinement_report, refinement_confusion = refinement_oracle.evaluate(
                eval_handler.feature_builder.get_vectors(), output_dict=True
            )
            refinement_results = dict()
            refinement_results["eval"] = refinement_report
            refinement_results["time"] = refinement_time
            refinement_results["confusion"] = refinement_confusion
            refinement_results["new"] = len(refinement_loop.new_feature_vectors)
            logging.info(f"Found {len(refinement_loop.new_feature_vectors)} new inputs")
            refinement_results["failing"] = len(
                list(
                    filter(
                        lambda vector: vector.result == TestResult.FAILING,
                        refinement_loop.new_feature_vectors,
                    )
                )
            )
            if refinement_loop.new_feature_vectors:
                report_new, confusion_new = oracle.evaluate(
                    refinement_loop.new_feature_vectors, output_dict=True
                )
                refinement_results["eval_new"] = report_new
                refinement_results["confusion_new"] = confusion_new
            result["refinement"][gens] = refinement_results

        results[i] = result

    with open(RESULTS_PATH / f"{project.get_identifier()}_refinement.json", "w") as fp:
        json.dump(results, fp, indent=2)


class Tests4PyEventCollector(EventCollector):
    def __init__(self, work_dir: os.PathLike, progress=False, split=True):
        super().__init__(work_dir)
        self.progress = progress
        self.split = split

    @staticmethod
    def convert(test_result: TestResult) -> TestResult:
        if test_result.name == "FAILING":
            return TestResult.FAILING
        else:
            return TestResult.PASSING

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
        for test in tqdm(tests) if self.progress else tests:
            test_input = test
            if isinstance(test, str):
                if self.split:
                    try:
                        test_input = shlex.split(test)
                    except ValueError:
                        pass
                else:
                    test_input = [test]
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
        handler: EventHandler,
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

    def run(self):
        for _ in tqdm(range(self.iterations)):
            self.iteration()
            self.all_features = self.features.all_features
        if self.new_feature_vectors:
            self.learned_oracle.finalize(self.new_feature_vectors)


def main(project_name: str, bug_id: int):
    project = t4p.get_projects(project_name, bug_id)[0]
    assert project.systemtests
    if not RESULTS_PATH.exists():
        RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    evaluate_project(project)


if __name__ == "__main__":
    random.seed(SEED)
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
