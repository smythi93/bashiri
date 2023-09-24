import random
import shutil
import string
import sys
import zipfile
from pathlib import Path
from typing import Tuple, List

import wget

DATA_URL = "https://github.com/githubhuyang/refactory/raw/master/data.zip"

REFACTORY = Path("refactory")
DATA = REFACTORY / "data"
QUESTION_1 = DATA / "question_1"
QUESTION_2 = DATA / "question_2"
QUESTION_3 = DATA / "question_3"
QUESTION_4 = DATA / "question_4"
QUESTION_5 = DATA / "question_5"

CODE = Path("code")
REFERENCE = CODE / "reference" / "reference.py"

EVAL = REFACTORY / "eval"
EVAL_QUESTION_1 = EVAL / "question_1"
EVAL_QUESTION_2 = EVAL / "question_2"
EVAL_QUESTION_3 = EVAL / "question_3"
EVAL_QUESTION_4 = EVAL / "question_4"
EVAL_QUESTION_5 = EVAL / "question_5"

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
MAX_DAYS = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def chance_half():
    return random.getrandbits(1)


def chance_quarter():
    return random.getrandbits(1) and random.getrandbits(1)


def get_data():
    file_name = wget.download(DATA_URL)
    with zipfile.ZipFile(file_name, "r") as zif_file:
        zif_file.extractall(REFACTORY)


def get_rand_int(start=-257, end=256) -> int:
    return random.randint(start, end)


def get_rand_sequence(producer, min_len=3, max_len=42):
    return [producer() for _ in range(random.randint(min_len, max_len))]


def get_rand_string(min_len=3, max_len=20) -> string:
    return "".join(random.choices(string.printable, k=random.randint(min_len, max_len)))


def get_rand_float(start=-257, end=256) -> float:
    return get_rand_int(start, end) + random.random()


def generate_question_1() -> str:
    if chance_half():
        producer = get_rand_int
    else:
        if chance_half():
            producer = get_rand_string
        else:
            producer = get_rand_float
    seq = sorted(get_rand_sequence(producer))
    if chance_half():
        seq = tuple(seq)
    value = producer()
    return f"search({repr(value)}, {repr(seq)})"


def get_rand_month() -> str:
    return random.choice(MONTHS)


def get_rand_day(month: str = None) -> str:
    if month:
        return random.randint(1, MAX_DAYS[MONTHS.index(month)])
    else:
        return random.randint(1, 31)


def get_rand_md() -> Tuple[str, int]:
    month = get_rand_month()
    return month, get_rand_day(month)


def get_rand_md_data(
    force_same_month=False, force_same_day=False
) -> List[Tuple[str, int]]:
    seq = get_rand_sequence(get_rand_md)
    if force_same_month or force_same_day:
        m, d = random.choice(seq)
        if not force_same_month:
            m = get_rand_month()
        if not force_same_day:
            d = get_rand_day(m)
        seq.append((m, d))
    random.shuffle(seq)
    return seq


def generate_question_2() -> str:
    function, producer = random.choice(
        [
            ("unique_day", get_rand_day),
            ("unique_month", get_rand_month),
            ("contains_unique_day", get_rand_month),
        ]
    )
    mds = get_rand_md_data(chance_quarter(), chance_quarter())
    if chance_half():
        mds = tuple(mds)
    return f"{function}({repr(producer())}, {repr(mds)})"


def generate_question_3() -> str:
    if chance_half():
        producer = get_rand_int
    else:
        if chance_half():
            producer = get_rand_string
        else:
            producer = get_rand_float
    seq = get_rand_sequence(producer, min_len=0)
    if seq and chance_quarter():
        seq += random.choices(seq, k=random.randint(1, min(5, len(seq))))
    random.shuffle(seq)
    return f"remove_extras({repr(seq)})"


def get_rand_tuple() -> Tuple[str, int]:
    if chance_quarter() and chance_quarter():
        g = "D"
    elif chance_half():
        g = "F"
    else:
        g = "M"
    return g, get_rand_int(0, 100)


def generate_question_4() -> str:
    seq = get_rand_sequence(get_rand_tuple, min_len=0)
    return f"sort_age({repr(seq)})"


def generate_question_5() -> str:
    seq = get_rand_sequence(get_rand_int, min_len=0)
    if seq:
        k = random.randint(0, len(seq))
    else:
        k = 0
    return f"top_k({repr(seq)}, {repr(k)})"


# noinspection PyPep8Naming
def generate_eval_examples(N: int = 400):
    places = max(3, len(str(N)))
    formatter = f"{{0:0{places}d}}"
    EVAL.mkdir(exist_ok=True)
    for q in (1, 2, 3, 4, 5):
        base: Path = globals()[f"EVAL_QUESTION_{q}"]
        base.mkdir(exist_ok=True)
        for n in range(1, N + 1):
            expression = globals()[f"generate_question_{q}"]()
            expected = eval(expression)
            with open(base / f"input_{formatter.format(n)}.txt", "w") as inp:
                inp.write(expression)
            with open(base / f"output_{formatter.format(n)}.txt", "w") as out:
                out.write(repr(expected))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        random.seed(int(sys.argv[1]))
    else:
        random.seed(42)
    if not REFACTORY.exists():
        get_data()
    for question in QUESTION_1, QUESTION_2, QUESTION_3, QUESTION_4, QUESTION_5:
        with open(question / REFERENCE, "r") as fp:
            exec(fp.read())
    shutil.rmtree(EVAL, ignore_errors=True)
    generate_eval_examples()
