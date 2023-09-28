class Confusion:
    def __init__(
        self,
        tp: int = 0,
        fn: int = 0,
        fp: int = 0,
        tn: int = 0,
        perfect: int = 0,
        total: int = 1,
        time: float = 0,
    ):
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn
        self.perfect = perfect
        self.total = total
        self.time = time

    def __add__(self, other):
        assert isinstance(other, Confusion)
        return Confusion(
            tp=self.tp + other.tp,
            fn=self.fn + other.fn,
            fp=self.fp + other.fp,
            tn=self.tn + other.tn,
            perfect=self.perfect + other.perfect,
            total=self.total + other.total,
            time=self.time + other.time,
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
        return (self.tp + self.tn) / max(self.total_labels(), 1) * 100

    def perfect_score(self) -> float:
        return self.perfect / max(self.total, 1) * 100

    def f1_bug(self) -> float:
        return 2 * self.tn / max(2 * self.tn + self.fn + self.fp, 1)

    def f1_no_bug(self) -> float:
        return 2 * self.tp / max(2 * self.tp + self.fp + self.fn, 1)

    def macro_precision(self):
        return (self.precision_bug() + self.precision_no_bug()) / 2

    def macro_recall(self):
        return (self.recall_bug() + self.recall_no_bug()) / 2

    def macro_f1(self):
        return (self.f1_bug() + self.f1_no_bug()) / 2

    def bugs(self):
        return self.tn + self.fp

    def no_bugs(self):
        return self.tp + self.fn

    def total_labels(self):
        return self.bugs() + self.no_bugs()

    def weighted_precision(self):
        return (
            self.bugs() * self.precision_bug()
            + self.no_bugs() * self.precision_no_bug()
        ) / max(self.total_labels(), 1)

    def weighted_recall(self):
        return (
            self.bugs() * self.recall_bug() + self.no_bugs() * self.recall_no_bug()
        ) / max(self.total_labels(), 1)

    def weighted_f1(self):
        return (self.bugs() * self.f1_bug() + self.no_bugs() * self.f1_no_bug()) / max(
            self.total_labels(), 1
        )

    def avg_time(self):
        return self.time / max(self.total, 1)

    def print(self):
        print(f"tp  : {self.tp}")
        print(f"fn  : {self.fn}")
        print(f"fp  : {self.fp}")
        print(f"tn  : {self.tn}")
        print(f"p   : {self.perfect}")
        print(f"t   : {self.total}")
        print(f"ac  : {self.accuracy():.2f}")
        print(f"pb  : {self.precision_bug():.4f}")
        print(f"pn  : {self.precision_no_bug():.4f}")
        print(f"rb  : {self.recall_bug():.4f}")
        print(f"rn  : {self.recall_no_bug():.4f}")
        print(f"f1b : {self.f1_bug():.4f}")
        print(f"f1n : {self.f1_no_bug():.4f}")
        print(f"mp  : {self.macro_precision():.4f}")
        print(f"mr  : {self.macro_recall():.4f}")
        print(f"mf1 : {self.macro_f1():.4f}")
        print(f"wap : {self.weighted_precision():.4f}")
        print(f"war : {self.weighted_recall():.4f}")
        print(f"waf1: {self.weighted_f1():.4f}")
        print(f"ps  : {self.perfect_score():.2f}")
        print(f"time: {self.avg_time():.2f}")


EVAL = "eval"
TIME = "time"
BUG = "1"
NO_BUG = "0"
CONFUSION = "confusion"


def get_confusion(dictionary: dict, name="") -> Confusion:
    result = Confusion(total=0)
    if CONFUSION not in dictionary:
        print(f"skip {name}: no {CONFUSION}")
        return result
    if EVAL not in dictionary:
        print(f"skip {name}: no {EVAL}")
        return result
    if TIME not in dictionary:
        print(f"skip {name}: no {TIME}")
        return result
    cm = dictionary[CONFUSION]
    if len(cm) == 1:
        if len(cm[0]) != 1:
            print(f"skip {name}: {CONFUSION} not correct format")
            return result
        if BUG in dictionary[EVAL]:
            result = Confusion(tn=cm[0][0], perfect=1)
        else:
            result = Confusion(tp=cm[0][0], perfect=1)
    else:
        tp = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tn = cm[1][1]
        result = Confusion(tp=tp, fp=fp, fn=fn, tn=tn, perfect=fp == 0 and fn == 0)
    result.time = dictionary[TIME]
    return result
