import argparse
import hashlib
import json
import logging.handlers
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Any, Dict, Tuple, List

import sflkit.logger
from sflkit.runners.run import TestResult
from tqdm import tqdm

from bashiri.events import EventCollector, instrument
from bashiri.features import EventHandler, FeatureBuilder
from bashiri.learning import DecisionTreeOracle

REFACTORY = Path("refactory")
DATA = REFACTORY / "data"
QUESTION_1 = DATA / "question_1"
QUESTION_2 = DATA / "question_2"
QUESTION_3 = DATA / "question_3"
QUESTION_4 = DATA / "question_4"
QUESTION_5 = DATA / "question_5"

EVAL = Path("refactory/eval")
EVAL_QUESTION_1 = EVAL / "question_1"
EVAL_QUESTION_2 = EVAL / "question_2"
EVAL_QUESTION_3 = EVAL / "question_3"
EVAL_QUESTION_4 = EVAL / "question_4"
EVAL_QUESTION_5 = EVAL / "question_5"

QUESTIONS: Dict[int, Tuple[Path, Path]] = {
    i: (globals()[f"QUESTION_{i}"], globals()[f"EVAL_QUESTION_{i}"])
    for i in range(1, 6)
}

CODE = Path("code")
REFERENCE = CODE / "reference" / "reference.py"
ANS = Path("ans")

ACCESS = "access.py"
DST = "tmp.py"

RESULTS_PATH = Path("results")

EXPECTED_OUTPUTS: Dict[int, Dict[str, Any]] = dict()
RESULTS: Dict[str, Dict[str, Any]] = dict()

FILE_PATTERN = re.compile(r"wrong_(?P<q>\d)_(?P<e>\d{3})\.py")
TIMEOUT: int = 2

LOGGER = logging.getLogger("refactory")
HANDLER = logging.handlers.WatchedFileHandler("refactory.log")
LOGGER.addHandler(HANDLER)
LOGGER.propagate = False

ONLY_FUNCTIONS = False

LIMIT = None


def get_features_from_tests(question: int, tests: Sequence[str]) -> FeatureBuilder:
    collector = RefactoryEventCollector(
        Path.cwd(), expected_results=EXPECTED_OUTPUTS.get(question, dict())
    )
    events = collector.get_events(tests)
    handler = EventHandler()
    handler.handle_files(events)
    return handler.feature_builder


def get_tests(question: int, path: Path, limit: Optional[int] = None) -> List[str]:
    if question not in EXPECTED_OUTPUTS:
        EXPECTED_OUTPUTS[question] = dict()
    # noinspection PyPep8Naming
    N = len(os.listdir(path)) // 2
    places = max(3, len(str(N)))
    formatter = f"{{0:0{places}d}}"
    tests = list()
    for n in range(1, (min(N, limit) if limit else N) + 1):
        with open(path / f"input_{formatter.format(n)}.txt", "r") as inp:
            test = inp.read()
        with open(path / f"output_{formatter.format(n)}.txt", "r") as out:
            expected = out.read()
        if test not in EXPECTED_OUTPUTS[question]:
            EXPECTED_OUTPUTS[question][test] = eval(expected)
        tests.append(test)
    return tests


def get_features(question: int, path: Path, limit: Optional[int] = None):
    return get_features_from_tests(question, get_tests(question, path, limit))


def get_model(question: int, ans_path):
    features = get_features(question, ans_path)
    all_features = features.get_all_features()
    path = Path("dt")
    if path.exists():
        os.remove(path)
    model = DecisionTreeOracle(path=path)
    model.fit(
        all_features,
        features.get_vectors(),
    )
    return model


def verify_example(
    question: int, file: Path, path: Path, limit: Optional[int] = None
) -> bool:
    tests = get_tests(question, path, limit=limit)
    shutil.copy(file, DST)
    results = [oracle(test, EXPECTED_OUTPUTS[question]) for test in tests]
    return TestResult.PASSING in results and TestResult.FAILING in results


def run_on_example(
    question: int, identifier: int, limit: Optional[int] = None, functions: bool = False
) -> Optional[Dict[str, Any]]:
    name = f"wrong_{question}_{identifier:03d}"
    path, eval_path = QUESTIONS[question]
    file: Path = path / CODE / "wrong" / f"{name}.py"
    if file.exists() and verify_example(question, file, path / ANS, limit=limit):
        LOGGER.info(f"Start evaluation of {name}")
        if functions:
            instrument(file, DST, events=["FUNCTION_ENTER"])
        else:
            instrument(file, DST)
        LOGGER.info(f"Get evaluation features of {name}")
        eval_features = get_features(question, eval_path, limit=limit)
        LOGGER.info(f"Get oracle for {name}")
        start = time.time()
        model = get_model(question, path / ANS)
        timing = time.time() - start
        report_eval, confusion_matrix = model.evaluate(
            eval_features.get_vectors(),
            output_dict=True,
        )
        result = {
            "eval": report_eval,
            "confusion": confusion_matrix,
            "time": timing,
        }
        RESULTS[name] = result
        return result
    else:
        LOGGER.info(f"Skip evaluation of {name}")


def run_on_question(
    question: int, limit: Optional[int] = None, functions: bool = False
):
    path, _ = QUESTIONS[question]
    directory: Path = path / CODE / "wrong"
    result = dict()
    if directory.exists():
        for file in tqdm(os.listdir(directory)):
            m = FILE_PATTERN.match(file)
            if m:
                q = int(m.group("q"))
                if q == question:
                    e = int(m.group("e"))
                    result[e] = run_on_example(
                        question, e, limit=limit, functions=functions
                    )
    return result


def parse_args(*args: str):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-q",
        "--question",
        dest="question",
        default=None,
        type=int,
        help="Evaluate on a single question",
    )
    arg_parser.add_argument(
        "-e",
        "--example",
        dest="example",
        default=None,
        type=int,
        help="Evaluate on a single example",
    )
    arg_parser.add_argument(
        "-l",
        "--limit",
        dest="limit",
        default=None,
        type=int,
        help="Limit for the number of inputs for the evaluation",
    )
    return arg_parser.parse_args(args or sys.argv[1:])


def main(*args: str, stdout=sys.stdout, stderr=sys.stderr):
    if stdout is not None:
        sys.stdout = stdout
    if stderr is not None:
        sys.stderr = stderr
    sflkit.logger.LOGGER.disabled = True
    args = parse_args(*args)
    if not RESULTS_PATH.exists():
        RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    result_file = "refactory"
    if args.question is None:
        for question in range(1, 6):
            run_on_question(
                question, limit=args.limit or LIMIT, functions=ONLY_FUNCTIONS
            )
    elif args.example is None:
        result_file += f"_{args.question}"
        run_on_question(
            args.question, limit=args.limit or LIMIT, functions=ONLY_FUNCTIONS
        )
    else:
        result_file += f"_{args.question}_{args.example}"
        run_on_example(
            args.question,
            args.example,
            limit=args.limit or LIMIT,
            functions=ONLY_FUNCTIONS,
        )
    with open(
        RESULTS_PATH / f"{result_file}{'_functions' if ONLY_FUNCTIONS else ''}.json",
        "w",
    ) as result_json:
        json.dump(RESULTS, result_json)


for question_x in QUESTION_1, QUESTION_2, QUESTION_3, QUESTION_4, QUESTION_5:
    with open(question_x / REFERENCE, "r") as fp:
        exec(fp.read())


def oracle(test: str, expected_results: Dict[str, Any] = None):
    expected_results = expected_results or dict()
    try:
        process = subprocess.run(
            ["python", ACCESS, test],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return TestResult.FAILING
    if process.returncode == 0:
        try:
            if test in expected_results:
                expected = expected_results[test]
            else:
                expected = eval(test)
            if expected == eval(process.stdout.decode("utf8")):
                return TestResult.PASSING
            else:
                return TestResult.FAILING
        except:
            return TestResult.FAILING
    else:
        return TestResult.FAILING


class RefactoryEventCollector(EventCollector):
    def __init__(self, work_dir: os.PathLike, expected_results: Dict[str, Any]):
        super().__init__(work_dir)
        self.expected_results = expected_results

    def collect(
        self,
        output: os.PathLike,
        tests: Optional[Sequence[str]] = None,
        label: Optional[TestResult] = None,
    ):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        for test_result in TestResult:
            (output / test_result.get_dir()).mkdir(parents=True, exist_ok=True)
        for test in tests:
            self.runs[test] = oracle(test, self.expected_results)
            if label is None:
                test_result = self.runs[test]
            else:
                test_result = label
            if os.path.exists(self.work_dir / "EVENTS_PATH"):
                shutil.move(
                    self.work_dir / "EVENTS_PATH",
                    output
                    / test_result.get_dir()
                    / hashlib.md5(test.encode("utf8")).hexdigest(),
                )


if __name__ == "__main__":
    if "-O" in sys.argv:
        sys.argv.remove("-O")
        os.execl(sys.executable, sys.executable, "-O", *sys.argv)
    else:
        main()
