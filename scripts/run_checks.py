"""Run the local full verification suite for the data and evaluation pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_config.example.json", help="Path to the shared data config JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = [
        [sys.executable, "-m", "unittest", "tests.test_data_evaluation"],
        [sys.executable, "scripts/sanity_check.py", "--config", args.config],
    ]

    for command in commands:
        print("Running:", " ".join(command))
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
