import subprocess
from pathlib import Path

DIR = Path(__file__).parent.resolve()

if __name__ == "__main__":
    subprocess.run(["python3", "-m", "pip", "install", "-e", "."])
    subprocess.run(["python3", "-m", "pip", "install", DIR / "sflkit-extension"])
    subprocess.run(["python3", "-m", "pip", "install", DIR / "sflkit-lib-extension"])
