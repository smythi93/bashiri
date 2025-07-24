import argparse
import hashlib
import json
import logging
import os
import random
import shlex
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Sequence, Any, Tuple

import sflkit.logger
import tests4py.api as t4p
import tests4py.logger
from sflkit.events.mapping import EventMapping
from sflkit.features.handler import EventHandler
from sflkit.runners.run import TestResult
from tests4py import sfl
from tests4py.api import run
from tests4py.projects import Project
from tqdm import tqdm

from bashiri.events import EventCollector
from bashiri import Bashiri, Mode
from bashiri.mapping.mapping import Mapping, MappingCreator
from bashiri.mapping.patch import PatchTranslator

TMP = Path("tmp_tests")
TEST_DIR = TMP / "test_dir"
EVENTS_DIR = TMP / "events"
EVENTS_TRAINING = EVENTS_DIR / "train"
EVENTS_EVALUATION = EVENTS_DIR / "eval"
EVENTS_FIXED = EVENTS_DIR / "fixed"
TRAINING = TMP / "train"
EVALUATION = TMP / "eval"
EVALUATION_FIXED = TMP / "eval_fixed"
FIXED_DIR = TMP / "fixed"
MAPPING = Path("mappings")

N = 200
F = 100
SEED = 42

REP = 5

N_MAPPING = 10
F_MAPPING = 5

RESULTS_PATH = Path("results")


@contextmanager
def fixed(project: Project):
    project.buggy = False
    yield
    project.buggy = True


def get_mapping(project: Project) -> Tuple[EventMapping, EventMapping, Mapping]:
    mapping_bug = MAPPING / f"{project}.json"
    with fixed(project):
        mapping_fix = MAPPING / f"{project}.json"
    if not mapping_bug.exists():
        raise FileNotFoundError(f"Mapping bug not found for {project}")
    if not mapping_fix.exists():
        raise FileNotFoundError(f"Mapping fix not found for {project}")
    mapping_bug = EventMapping.load_from_file(mapping_bug, "")
    mapping_fix = EventMapping.load_from_file(mapping_fix, "")

    patch = PatchTranslator.build_t4p_translator(project)
    creator = MappingCreator(mapping_bug)
    mapping = creator.create(mapping_fix, patch)
    mapping.dump(MAPPING / f"{project.get_identifier()}_translation.json", indent=1)
    translation_mapping = EventMapping(
        mapping_bug.mapping,
        translation=mapping.get_translation(),
        alternative_mapping=mapping_fix.mapping,
    )
    return translation_mapping


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
    if report.raised:
        raise report.raised

    with fixed(project):
        report = t4p.checkout(project)
        if report.raised:
            raise report.raised
        fixed_mapping_path = MAPPING / f"{project}.json"
        report = sfl.sflkit_instrument(FIXED_DIR, location, mapping=fixed_mapping_path)
        if report.raised:
            raise report.raised
        report = t4p.systemtest_generate(
            FIXED_DIR, EVALUATION_FIXED, n=N_MAPPING, p=F_MAPPING
        )
        if report.raised:
            raise report.raised

    translation = get_mapping(project)

    eval_inputs = []
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
    time.sleep(1)
    eval_handler = EventHandler()
    logging.info(f"Constructing features")
    for e in tqdm(eval_events):
        eval_handler.handle(e)
    assert len(eval_handler.builder.feature_vectors) == len(eval_events)

    with fixed(project):
        mapping_inputs = []
        for file in os.listdir(EVALUATION_FIXED):
            with open(EVALUATION_FIXED / file, "r") as fp:
                mapping_inputs.append(fp.read())
        # do not split input with shlex for cookiecutter
        mapping_collector = Tests4PyEventCollector(
            FIXED_DIR,
            location,
            progress=True,
            split=project.project_name != "cookiecutter",
            mapping=translation,
        )
        logging.info(f"Collecting events for fixed version")
        mapping_events = mapping_collector.get_events(mapping_inputs)
        mapping_handler = EventHandler()
        logging.info(f"Constructing features for fixed version")
        for e in tqdm(mapping_events):
            mapping_handler.handle(e)
        assert len(mapping_handler.builder.feature_vectors) == len(mapping_events)

    results = dict()

    logging.info(f"Running configurations")
    for i in range(REP):
        for t, f in ((5, 1), (5, 2), (10, 1), (10, 2), (20, 2), (30, 3)):
            logging.info(f"Starting evaluation of {t} tests and {f} failing")
            result = dict()

            report = t4p.systemtest_generate(TEST_DIR, TRAINING, n=t, p=f)
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
            path = Path("../dt")
            if path.exists():
                os.remove(path)
            bashiri = Bashiri(
                path=path,
                tests=inputs,
                random_state=SEED,
                collector=collector,
                mode=Mode.CUSTOM,
            )
            bashiri.learn()
            obe_time = time.time() - obe_time
            report_eval, confusion = bashiri.evaluate(
                eval_handler,
                output_dict=True,
            )
            mapping_eval, mapping_confusion = bashiri.evaluate(
                mapping_handler,
                output_dict=True,
            )
            result["eval"] = report_eval
            result["mapping_eval"] = mapping_eval
            result["time"] = obe_time
            result["confusion"] = confusion
            result["mapping_confusion"] = mapping_confusion

            results[f"tests_{i}_{t}_failing_{f}"] = result

    with open(RESULTS_PATH / f"{project.get_identifier()}.json", "w") as fp:
        json.dump(results, fp, indent=2)


class Tests4PyEventCollector(EventCollector):
    def __init__(
        self,
        work_dir: os.PathLike,
        src: os.PathLike,
        progress=False,
        split=True,
        mapping=None,
    ):
        super().__init__(work_dir, src, mapping=mapping)
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
                report = run(self.work_dir, test_input, invoke_oracle=True)
                if report.raised:
                    test_result = TestResult.UNDEFINED
                else:
                    test_result = self.convert(report.test_result)
            else:
                report = run(self.work_dir, test_input)
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
