import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import sflkit.logger
import tests4py.api as t4p
import tests4py.logger
from sflkit.features.handler import EventHandler
from sflkit.features.vector import FeatureVector
from sflkit.runners.run import TestResult
from tests4py import sfl
from tests4py.projects import Project
from tqdm import tqdm

from bashiri.events import EventCollector
from bashiri.learning import Bashiri, Oracle, Label, CausalTree
from bashiri.refinement import (
    DifferenceInterestRefinement,
)
from run_on_tests4py import Tests4PyEventCollector, N, F, SEED, MAPPING

TMP = Path("tmp_tests")
TEST_DIR = TMP / "test_dir"
EVENTS_DIR = TMP / "events"
EVENTS_TRAINING = EVENTS_DIR / "train"
EVENTS_EVALUATION = EVENTS_DIR / "eval"
TRAINING = TMP / "train"
EVALUATION = TMP / "eval"

RLS = [50]

RESULTS_PATH = Path("results")


def evaluate_project(project: Project):
    project.buggy = True
    logging.info(f"Checking out subject {project}")
    report = t4p.checkout(project)
    if report.raised:
        raise report.raised
    location = Path(report.location)
    logging.info(f"Instrumenting and compiling subject {project}")
    MAPPING.mkdir(parents=True, exist_ok=True)
    mapping_path = MAPPING / f"{project}.json"
    report = sfl.sflkit_instrument(TEST_DIR, location, mapping=mapping_path)
    if report.raised:
        raise report.raised
    logging.info(f"Generating {N} tests ({F} failing) for evaluation")
    report = t4p.systemtest_generate(TEST_DIR, EVALUATION, n=N, p=F)
    eval_inputs = []
    if report.raised:
        raise report.raised
    for file in os.listdir(EVALUATION):
        with open(EVALUATION / file, "r") as fp:
            eval_inputs.append(fp.read())
    # do not split input with shlex for cookiecutter
    eval_collector = Tests4PyEventCollector(
        TEST_DIR,
        location,
        progress=True,
        split=project.project_name != "cookiecutter",
        mapping=mapping_path,
    )
    logging.info(f"Collecting events")
    eval_events = eval_collector.get_events(eval_inputs)
    eval_handler = EventHandler()
    logging.info(f"Constructing features")
    for e in tqdm(eval_events):
        eval_handler.handle(e)
    assert len(eval_handler.builder.feature_vectors) == len(eval_events)

    results = dict()

    logging.info(f"Running configurations")
    for i in range(3):
        logging.info(f"Starting evaluation of {i}")
        result = dict()

        report = t4p.systemtest_generate(TEST_DIR, TRAINING, n=2, p=1)
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
            TEST_DIR,
            location,
            progress=True,
            split=project.project_name != "cookiecutter",
            mapping=mapping_path,
        )
        events = collector.get_events(inputs)
        handler = EventHandler()
        handler.handle_files(events)
        all_features = handler.builder.get_all_features()
        path = Path("../dt")
        if path.exists():
            os.remove(path)
        oracle = Bashiri(handler, CausalTree(), events)
        oracle.fit(
            all_features,
            handler,
        )
        obe_time = time.time() - obe_time
        report_eval, confusion = oracle.evaluate(
            eval_handler,
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
            refinement_oracle = Bashiri(handler, CausalTree(), events)
            refinement_oracle.fit(all_features, handler)
            refinement_time = time.time()
            refinement_loop = Tests4PyEvaluationRefinement(
                handler.copy(),
                refinement_oracle,
                {i: args for i, args in enumerate(inputs)},
                # do not split input with shlex for cookiecutter
                Tests4PyEventCollector(
                    TEST_DIR,
                    location,
                    progress=False,
                    split=project.project_name != "cookiecutter",
                    mapping=mapping_path,
                ),
                gens=gens,
            )
            refinement_loop.run()
            refinement_time = time.time() - refinement_time
            refinement_report, refinement_confusion = refinement_oracle.evaluate(
                eval_handler, output_dict=True
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
                report_new, confusion_new = oracle.evaluate_vectors(
                    refinement_loop.new_feature_vectors, output_dict=True
                )
                refinement_results["eval_new"] = report_new
                refinement_results["confusion_new"] = confusion_new
            result["refinement"][gens] = refinement_results

        results[i] = result

    with open(RESULTS_PATH / f"{project.get_identifier()}_refinement.json", "w") as fp:
        json.dump(results, fp, indent=2)


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
