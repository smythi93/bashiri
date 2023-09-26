import json
from pathlib import Path

RESULTS_1 = Path("refactory_1.json")
RESULTS_2 = Path("refactory_2.json")
RESULTS_3 = Path("refactory_3.json")
RESULTS_4 = Path("refactory_4.json")
RESULTS_5 = Path("refactory_5.json")

EVAL = "eval"
BUG = "1"
NO_BUG = "0"
CONFUSION = "confusion"


class Confusion:
    def __init__(
        self,
        tp: int = 0,
        fn: int = 0,
        fp: int = 0,
        tn: int = 0,
        perfect: int = 0,
        total: int = 1,
    ):
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn
        self.perfect = perfect
        self.total = total

    def __add__(self, other):
        assert isinstance(other, Confusion)
        return Confusion(
            self.tp + other.tp,
            self.fp + other.fp,
            self.fn + other.fn,
            self.tn + other.tn,
            self.perfect + other.perfect,
            self.total + other.total,
        )

    def precision_bug(self) -> float:
        return self.tn / max(self.tn + self.fn, 1)

    def precision_no_bug(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    def recall_bug(self) -> float:
        return self.tn / max(self.tn + self.fp, 1)

    def recall_no_bug(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    def accuracy(self) -> float:
        return (self.tp + self.tn) / max(self.tp + self.fp + self.fn + self.tn, 1)

    def perfect_score(self) -> float:
        return self.perfect / max(self.total, 1)

    def f1_bug(self) -> float:
        return 2 * self.tn / (2 * self.tn + self.fn + self.fp)

    def f1_no_bug(self) -> float:
        return 2 * self.tp / (2 * self.tp + self.fp + self.fn)

    def print(self):
        print(f"tp: {self.tp}")
        print(f"fp: {self.fp}")
        print(f"fn: {self.tn}")
        print(f"tn: {self.tn}")
        print(f"p : {self.perfect}")
        print(f"t : {self.total}")
        print(f"ac: {self.accuracy()}")
        print(f"pb: {self.precision_bug()}")
        print(f"pn: {self.precision_no_bug()}")
        print(f"rb: {self.recall_bug()}")
        print(f"rn: {self.recall_no_bug()}")
        print(f"fb: {self.f1_bug()}")
        print(f"fn: {self.f1_no_bug()}")
        print(f"ps: {self.perfect_score()}")


def get_results(path: Path) -> Confusion:
    result = Confusion(total=0)
    if path.exists():
        refactory_results = json.loads(path.read_text("utf8"))
        for name in refactory_results:
            if CONFUSION not in refactory_results[name]:
                print(f"skip {name}")
                continue
            if EVAL not in refactory_results[name]:
                print(f"skip {name}")
                continue
            cm = refactory_results[name][CONFUSION]
            if len(cm) == 1:
                if len(cm[0]) != 1:
                    print(f"skip {name}")
                    continue
                if BUG in refactory_results[name][EVAL]:
                    cm = Confusion(tn=cm[0][0], perfect=1)
                else:
                    cm = Confusion(tp=cm[0][0], perfect=1)
            else:
                tp = cm[0][0]
                fp = cm[0][1]
                fn = cm[1][0]
                tn = cm[1][1]
                cm = Confusion(tp=tp, fp=fp, fn=fn, tn=tn, perfect=fp == 0 and fn == 0)
            result += cm
    return result


def main():
    results = Confusion(total=0)
    for path in (RESULTS_1, RESULTS_2, RESULTS_3, RESULTS_4, RESULTS_5):
        results += get_results(path)
    results.print()


if __name__ == "__main__":
    main()
