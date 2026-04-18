"""
update_data.py
Entry point: fetch new match results, then re-run the ELO pipeline.

Usage:
    python update_data.py
"""

import subprocess
import sys


def run(script: str) -> bool:
    print(f"\n{'='*60}")
    print(f"Running {script}...")
    print('='*60)
    result = subprocess.run([sys.executable, script], check=False)
    if result.returncode != 0:
        print(f"\nERROR: {script} exited with code {result.returncode}")
        return False
    return True


if __name__ == "__main__":
    if not run("fetch_results.py"):
        sys.exit(1)

    print("\nDone. Open main.ipynb and run all cells to recalculate ELO ratings.")
