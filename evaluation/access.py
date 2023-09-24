import sys

from tmp import *

if __name__ == "__main__":
    assert len(sys.argv) == 2
    print(repr(eval(sys.argv[1])))
