import os
import sys

import run_on_refactory

if __name__ == "__main__":
    if "-O" in sys.argv:
        sys.argv.remove("-O")
        os.execl(sys.executable, sys.executable, "-O", *sys.argv)
    else:
        run_on_refactory.ONLY_FUNCTIONS = True
        run_on_refactory.TIMEOUT = 0.5
        run_on_refactory.main()
