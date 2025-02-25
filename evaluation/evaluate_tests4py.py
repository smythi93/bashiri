import json
import sys
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt

from confusion import (
    Confusion,
    get_confusion,
    EVAL,
    CONFUSION,
    TIME,
    MAPPING_CONFUSION,
    MAPPING_EVAL,
)

RESULTS = Path("results")

MIDDLE_1 = RESULTS / "middle_1.json"
MIDDLE_2 = RESULTS / "middle_2.json"
MARKUP_1 = RESULTS / "markup_1.json"
MARKUP_2 = RESULTS / "markup_2.json"
COOKIECUTTER_2 = RESULTS / "cookiecutter_2.json"
COOKIECUTTER_3 = RESULTS / "cookiecutter_3.json"
COOKIECUTTER_4 = RESULTS / "cookiecutter_4.json"
FASTAPI_1 = RESULTS / "fastapi_1.json"
PYSNOOPER_2 = RESULTS / "pysnooper_2.json"
PYSNOOPER_3 = RESULTS / "pysnooper_3.json"

REFINEMENT = "refinement"
NEW_EVAL = f"{EVAL}_new"
NEW_CONFUSION = f"{CONFUSION}_new"


def get_results(
    path: Path, refinement: bool = False
) -> Tuple[
    Confusion, Confusion, Confusion, Confusion, List[float], List[float], List[float]
]:
    results = Confusion(total=0)
    results_mapping = Confusion(total=0)
    results_refinement = Confusion(total=0)
    results_new_tests = Confusion(total=0)
    accuracies = list()
    precisions = list()
    recalls = list()
    if path.exists():
        tests4py_results = json.loads(path.read_text("utf8"))
        for config in tests4py_results:
            tmp_results = get_confusion(tests4py_results[config], f"{path}_{config}")
            accuracies.append(tmp_results.accuracy())
            precisions.append(tmp_results.precision_bug())
            recalls.append(tmp_results.recall_bug())
            results += tmp_results
            tmp_mapping_results = get_confusion(
                tests4py_results[config],
                f"{path}_{config}_mapping",
                conf=MAPPING_CONFUSION,
                ev=MAPPING_EVAL,
            )
            results_mapping += tmp_mapping_results
            if refinement:
                for refinement_conf in tests4py_results[config][REFINEMENT]:
                    results_refinement += get_confusion(
                        tests4py_results[config][REFINEMENT][refinement_conf],
                        f"{path}_{config}_{REFINEMENT}_{refinement}",
                    )
                    if (
                        NEW_EVAL
                        in tests4py_results[config][REFINEMENT][refinement_conf]
                        and NEW_CONFUSION
                        in tests4py_results[config][REFINEMENT][refinement_conf]
                    ):
                        results_new_tests += get_confusion(
                            {
                                EVAL: tests4py_results[config][REFINEMENT][
                                    refinement_conf
                                ][NEW_EVAL],
                                CONFUSION: tests4py_results[config][REFINEMENT][
                                    refinement_conf
                                ][NEW_CONFUSION],
                                TIME: 0,
                            },
                            f"{path}_{config}_new_{refinement}",
                            exclude_no_eval=False,
                        )
    return (
        results,
        results_mapping,
        results_refinement,
        results_new_tests,
        accuracies,
        precisions,
        recalls,
    )


def main(refinement: bool = False):
    results = Confusion(total=0)
    results_mapping = Confusion(total=0)
    results_refinement = Confusion(total=0)
    results_new_tests = Confusion(total=0)
    accuracies = dict()
    precisions = dict()
    recalls = dict()
    per_subject = dict()
    for path in [
        MIDDLE_1,
        MIDDLE_2,
        FASTAPI_1,
        PYSNOOPER_2,
        PYSNOOPER_3,
        COOKIECUTTER_4,
        COOKIECUTTER_3,
        COOKIECUTTER_2,
    ]:
        if refinement:
            parts = path.parts
            path = Path(*parts[:-1], parts[-1].replace(".json", "_refinement.json"))
        r, rm, rr, rnt, as_, ps_, rs_ = get_results(path, refinement=refinement)
        results += r
        results_mapping += rm
        results_refinement += rr
        results_new_tests += rnt
        name = path.parts[-1].split(".")[0]
        accuracies[name] = as_
        precisions[name] = ps_
        recalls[name] = rs_
        per_subject[name] = r, rr

    print("Results:")
    results.print()
    if not refinement:
        print()
        print("Results mapping:")
        results_mapping.print()

    if refinement:
        print()
        print("Results refinement:")
        results_refinement.print()
        print()
        print("Results new tests:")
        results_new_tests.print()
        for n in per_subject:
            r, rr = per_subject[n]
            print()
            print(f"{n}: {r.accuracy():.1f} -- {rr.accuracy():.1f} ")
            print(
                f"{n}: {r.average_macro_precision():.2f} -- {rr.average_macro_precision():.2f} "
            )
            print(
                f"{n}: {r.average_macro_recall():.2f} -- {rr.average_macro_recall():.2f} "
            )
            print(f"{n}: {r.average_macro_f1():.2f} -- {rr.average_macro_f1():.2f} ")
            print(f"{n}: {r.avg_time():.1f} -- {rr.avg_time():.1f} ")
    else:
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
    main(len(sys.argv) > 1 and sys.argv[1] == "-r")
