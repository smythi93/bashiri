import json
import sys
from pathlib import Path
from confusion import Confusion, get_confusion

RESULTS_1 = Path("results", "refactory_1.json")
RESULTS_2 = Path("results", "refactory_2.json")
RESULTS_3 = Path("results", "refactory_3.json")
RESULTS_4 = Path("results", "refactory_4.json")
RESULTS_5 = Path("results", "refactory_5.json")

EVAL = "eval"
TIME = "time"
BUG = "1"
NO_BUG = "0"
CONFUSION = "confusion"


def get_results(path: Path) -> Confusion:
    result = Confusion(total=0)
    if path.exists():
        refactory_results = json.loads(path.read_text("utf8"))
        for name in refactory_results:
            result += get_confusion(refactory_results[name], name=name)
    return result


def main(function: bool = False):
    results = Confusion(total=0)
    for path in (RESULTS_1, RESULTS_2, RESULTS_3, RESULTS_4, RESULTS_5):
        if function:
            parts = path.parts
            path = Path(*parts[:-1], parts[-1].replace(".json", "_functions.json"))
        results += get_results(path)
    results.print()


if __name__ == "__main__":
    main(len(sys.argv) > 1 and sys.argv[1] == "-f")
