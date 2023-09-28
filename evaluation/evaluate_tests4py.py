import json
import sys
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt

from confusion import Confusion, get_confusion, EVAL, CONFUSION, TIME

RESULTS = Path("results")

MIDDLE_1 = RESULTS / "middle_1.json"
COOKIECUTTER_2 = RESULTS / "cookiecutter_2.json"
COOKIECUTTER_3 = RESULTS / "cookiecutter_3.json"
COOKIECUTTER_4 = RESULTS / "cookiecutter_4.json"
FASTAPI_1 = RESULTS / "fastapi_1.json"
FASTAPI_2 = RESULTS / "fastapi_2.json"
PYSNOOPER_2 = RESULTS / "pysnooper_2.json"
PYSNOOPER_3 = RESULTS / "pysnooper_3.json"

REFINEMENT = "refinement"
NEW_EVAL = f"{EVAL}_new"
NEW_CONFUSION = f"{CONFUSION}_new"


def get_results(
    path: Path,
) -> Tuple[Confusion, Confusion, Confusion, List[float], List[float], List[float]]:
    results = Confusion(total=0)
    results_refinement = Confusion(total=0)
    results_new_tests = Confusion(total=0)
    accuracies = list()
    precisions = list()
    recalls = list()
    if path.exists():
        refactory_results = json.loads(path.read_text("utf8"))
        for config in refactory_results:
            tmp_results = get_confusion(refactory_results[config], f"{path}_{config}")
            accuracies.append(tmp_results.accuracy())
            precisions.append(tmp_results.precision_bug())
            recalls.append(tmp_results.recall_bug())
            results += tmp_results
            if REFINEMENT in refactory_results[config]:
                for refinement in refactory_results[config][REFINEMENT]:
                    results_refinement += get_confusion(
                        refactory_results[config][REFINEMENT][refinement],
                        f"{path}_{config}_{REFINEMENT}_{refinement}",
                    )
                    if (
                        NEW_EVAL in refactory_results[config][REFINEMENT][refinement]
                        and NEW_CONFUSION
                        in refactory_results[config][REFINEMENT][refinement]
                    ):
                        results_new_tests += get_confusion(
                            {
                                EVAL: refactory_results[config][REFINEMENT][refinement][
                                    NEW_EVAL
                                ],
                                CONFUSION: refactory_results[config][REFINEMENT][
                                    refinement
                                ][NEW_CONFUSION],
                                TIME: 0,
                            },
                            f"{path}_{config}_new_{refinement}",
                        )
    return (
        results,
        results_refinement,
        results_new_tests,
        accuracies,
        precisions,
        recalls,
    )


def main(function=False):
    results = Confusion(total=0)
    results_refinement = Confusion(total=0)
    results_new_tests = Confusion(total=0)
    accuracies = dict()
    precisions = dict()
    recalls = dict()
    for path in [
        MIDDLE_1,
        FASTAPI_1,
        PYSNOOPER_2,
        PYSNOOPER_3,
        COOKIECUTTER_4,
        COOKIECUTTER_3,
        COOKIECUTTER_2,
        FASTAPI_2,
    ]:
        if function:
            parts = path.parts
            path = Path(*parts[:-1], parts[-1].replace(".json", "_functions.json"))
        r, rr, rnt, as_, ps_, rs_ = get_results(path)
        results += r
        results_refinement += rr
        results_new_tests += rnt
        name = path.parts[-1].split(".")[0]
        accuracies[name] = as_
        precisions[name] = ps_
        recalls[name] = rs_

    print("Results:")
    results.print()
    if function:
        return
    print()
    print("Results refinement:")
    results_refinement.print()
    print()
    print("Results new tests:")
    results_new_tests.print()
    as_labels, as_values = zip(
        *[(name.replace("_", "."), accuracies[name]) for name in accuracies]
    )
    ps_labels, ps_values = zip(
        *[(name.replace("_", "."), precisions[name]) for name in precisions]
    )
    rs_labels, rs_values = zip(
        *[(name.replace("_", "."), recalls[name]) for name in recalls]
    )
    size = 14
    plt.rc("xtick", labelsize=size)
    plt.rc("ytick", labelsize=size)
    plt.figure(figsize=(8, 5))
    plt.boxplot(as_values)
    plt.xlabel("Subject", fontsize=size)
    plt.xticks(
        list(range(1, len(as_labels) + 1)),
        as_labels,
        rotation=-30,
        ha="left",
        rotation_mode="anchor",
    )
    plt.ylabel("Accuracy", fontsize=size)
    plt.tight_layout()
    plt.savefig("accuracies.pdf")
    plt.clf()
    plt.boxplot(ps_values)
    plt.xlabel("Subject", fontsize=size)
    plt.xticks(
        list(range(1, len(ps_labels) + 1)),
        ps_labels,
        rotation=-30,
        ha="left",
        rotation_mode="anchor",
    )
    plt.ylabel("Precision", fontsize=size)
    plt.savefig("precisions.pdf")
    plt.clf()
    plt.boxplot(rs_values)
    plt.xlabel("Subject", fontsize=size)
    plt.xticks(
        list(range(1, len(rs_labels) + 1)),
        rs_labels,
        rotation=-30,
        ha="left",
        rotation_mode="anchor",
    )
    plt.ylabel("Recall", fontsize=size)
    plt.savefig("recalls.pdf")
    plt.clf()


if __name__ == "__main__":
    main(len(sys.argv) > 1 and sys.argv[1] == "-f")
