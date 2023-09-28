import numpy as np
from sklearn.metrics import auc


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
        final: bool = False,
    ):
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn
        self.perfect = perfect
        self.total = total
        self.time = time
        self.all_confusions = [self] if final else []

    def __add__(self, other):
        assert isinstance(other, Confusion)
        all_confusions = self.all_confusions + other.all_confusions
        confusion = Confusion(
            tp=self.tp + other.tp,
            fn=self.fn + other.fn,
            fp=self.fp + other.fp,
            tn=self.tn + other.tn,
            perfect=self.perfect + other.perfect,
            total=self.total + other.total,
            time=self.time + other.time,
        )
        confusion.all_confusions = all_confusions
        return confusion

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

    def auc_bug(self):
        return auc(self.tn / (self.tn + self.fp), self.fn / (self.fn + self.tp))

    def auc_no_bug(self):
        return auc(self.tp / (self.tp + self.fn), self.fp / (self.fp + self.tn))

    def macro_auc(self):
        return (self.auc_bug() + self.auc_no_bug()) / 2

    def bugs(self):
        return self.tn + self.fp

    def no_bugs(self):
        return self.tp + self.fn

    def total_labels(self):
        return self.bugs() + self.no_bugs()

    def avg_time(self):
        return self.time / max(self.total, 1)

    def average_precision_bug(self) -> float:
        return np.average(list(map(Confusion.precision_bug, self.all_confusions)))

    def average_precision_no_bug(self) -> float:
        return np.average(list(map(Confusion.precision_no_bug, self.all_confusions)))

    def average_recall_bug(self) -> float:
        return np.average(list(map(Confusion.recall_bug, self.all_confusions)))

    def average_recall_no_bug(self) -> float:
        return np.average(list(map(Confusion.recall_no_bug, self.all_confusions)))

    def average_accuracy(self) -> float:
        return np.average(list(map(Confusion.accuracy, self.all_confusions)))

    def average_f1_bug(self) -> float:
        return np.average(list(map(Confusion.f1_bug, self.all_confusions)))

    def average_f1_no_bug(self) -> float:
        return np.average(list(map(Confusion.f1_no_bug, self.all_confusions)))

    def average_macro_precision(self):
        return np.average(list(map(Confusion.macro_precision, self.all_confusions)))

    def average_macro_recall(self):
        return np.average(list(map(Confusion.macro_recall, self.all_confusions)))

    def average_macro_f1(self):
        return np.average(list(map(Confusion.macro_f1, self.all_confusions)))

    def average_macro_auc(self):
        return np.average(list(map(Confusion.macro_auc, self.all_confusions)))

    def average_auc_bug(self):
        return np.average(list(map(Confusion.auc_bug, self.all_confusions)))

    def average_auc_no_bug(self):
        return np.average(list(map(Confusion.auc_no_bug, self.all_confusions)))

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
        print(f"ps  : {self.perfect_score():.2f}")
        print(f"time: {self.avg_time():.2f}")
        print()
        print(f"aac  : {self.average_accuracy():.2f}")
        print(f"apb  : {self.average_precision_bug():.4f}")
        print(f"apn  : {self.average_precision_no_bug():.4f}")
        print(f"arb  : {self.average_recall_bug():.4f}")
        print(f"arn  : {self.average_recall_no_bug():.4f}")
        print(f"af1b : {self.average_f1_bug():.4f}")
        print(f"af1n : {self.average_f1_no_bug():.4f}")
        print(f"amp  : {self.average_macro_precision():.4f}")
        print(f"amr  : {self.average_macro_recall():.4f}")
        print(f"amf1 : {self.average_macro_f1():.4f}")


EVAL = "eval"
TIME = "time"
BUG = "1"
NO_BUG = "0"
CONFUSION = "confusion"


def get_confusion(dictionary: dict, name="", exclude_no_eval=True) -> Confusion:
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
        if exclude_no_eval:
            return result
        if len(cm[0]) != 1:
            print(f"skip {name}: {CONFUSION} not correct format")
            return result
        if BUG in dictionary[EVAL]:
            result = Confusion(tn=cm[0][0], perfect=1, final=True)
        else:
            result = Confusion(tp=cm[0][0], perfect=1, final=True)
    else:
        tp = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tn = cm[1][1]
        result = Confusion(
            tp=tp, fp=fp, fn=fn, tn=tn, perfect=fp == 0 and fn == 0, final=True
        )
    result.time = dictionary[TIME]
    return result
