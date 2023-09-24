import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Any, Dict, Tuple

from sflkit.runners.run import TestResult

from bashiri.events import EventCollector, instrument
from bashiri.features import Handler, FeatureBuilder
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


EXPECTED_OUTPUTS: Dict[int, Dict[str, Any]] = dict()

RESULTS: Dict[str, Dict[str, Any]] = dict()

FILE_PATTERN = re.compile(r"wrong_(?P<q>\d)_(?P<e>\d{3})\.py")


def get_features_from_tests(question: int, tests: Sequence[str]) -> FeatureBuilder:
    collector = RefactoryEventCollector(
        Path.cwd(), expected_results=EXPECTED_OUTPUTS.get(question, dict())
    )
    events = collector.get_events(tests)
    handler = Handler()
    handler.handle_files(events)
    return handler.feature_builder


def get_features(question: int, path: Path, limit: Optional[int] = None):
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
    return get_features_from_tests(question, tests)


def get_model(question: int, ans_path):
    features = get_features(question, ans_path)
    all_features = features.get_all_features()
    path = Path("dt")
    if path.exists():
        os.remove(path)
    oracle = DecisionTreeOracle(path=path)
    oracle.fit(
        all_features,
        features.get_vectors(),
    )
    return oracle


def run_on_example(
    question: int, identifier: int, limit: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    name = f"wrong_{question}_{identifier:03d}"
    path, eval_path = QUESTIONS[question]
    file: Path = path / CODE / "wrong" / f"{name}.py"
    if file.exists():
        logging.info(f"Start evaluation of {name}")
        instrument(file, DST)
        logging.info(f"Get evaluation features of {name}")
        eval_features = get_features(question, eval_path, limit=limit)
        logging.info(f"Get oracle for {name}")
        start = time.time()
        oracle = get_model(question, path / ANS)
        timing = time.time() - start
        report_eval, confusion_matrix = oracle.evaluate(
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


def run_on_question(question: int, limit: Optional[int] = None):
    path, _ = QUESTIONS[question]
    directory: Path = path / CODE / "wrong"
    result = dict()
    if directory.exists():
        for file in os.listdir(directory):
            m = FILE_PATTERN.match(file)
            if m:
                q = int(m.group("q"))
                if q == question:
                    e = int(m.group("e"))
                    result[e] = run_on_example(question, e, limit=limit)
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
    args = parse_args(*args)
    result_file = "refactory"
    if args.question is None:
        for question in range(1, 6):
            run_on_question(question, limit=args.limit)
    elif args.example is None:
        result_file += f"_{args.question}"
        run_on_question(args.question, limit=args.limit)
    else:
        result_file += f"_{args.question}_{args.example}"
        run_on_example(args.question, args.example, limit=args.limit)
    with open(f"{result_file}.json", "w") as result_json:
        json.dump(RESULTS, result_json)


for question_x in QUESTION_1, QUESTION_2, QUESTION_3, QUESTION_4, QUESTION_5:
    with open(question_x / REFERENCE, "r") as fp:
        exec(fp.read())


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
            process = subprocess.run(
                ["python", ACCESS, test],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            if process.returncode == 0:
                try:
                    if test in self.expected_results:
                        expected = self.expected_results[test]
                    else:
                        expected = eval(test)
                    if expected == eval(process.stdout.decode("utf8")):
                        self.runs[test] = TestResult.PASSING
                    else:
                        self.runs[test] = TestResult.FAILING
                except:
                    self.runs[test] = TestResult.FAILING
            else:
                self.runs[test] = TestResult.FAILING
            if label is None:
                test_result = self.runs[test]
            else:
                test_result = label
            if os.path.exists(self.work_dir / "EVENTS_PATH"):
                shutil.move(
                    self.work_dir / "EVENTS_PATH",
                    output
                    / test_result.get_dir()
                    / hashlib.md5(" ".join(test).encode("utf8")).hexdigest(),
                )


if __name__ == "__main__":
    if "-O" in sys.argv:
        sys.argv.remove("-O")
        os.execl(sys.executable, sys.executable, "-O", *sys.argv)
    else:
        main()
