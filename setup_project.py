"""Create the Smart IV Infusion Monitor project skeleton.

Safe to run multiple times: existing files are never overwritten.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DIRECTORIES = [
    "config",
    "data/raw",
    "data/processed",
    "src/simulation",
    "src/preprocessing",
    "src/ekf",
    "src/prediction",
    "dashboard",
    "tests",
    "notebooks",
    "results",
]

# Empty files that will be filled in during later steps.
PLACEHOLDER_FILES = [
    "README.md",
    "config/config.yaml",
    "src/__init__.py",
    "src/simulation/__init__.py",
    "src/simulation/generate_data.py",
    "src/simulation/load_cell.py",
    "src/simulation/drop_sensor.py",
    "src/preprocessing/__init__.py",
    "src/preprocessing/processor.py",
    "src/ekf/__init__.py",
    "src/ekf/ekf.py",
    "src/prediction/__init__.py",
    "src/prediction/remaining_time.py",
    "src/prediction/uncertainty.py",
    "src/pipeline.py",
    "dashboard/app.py",
    "dashboard/components.py",
    "dashboard/charts.py",
    "tests/__init__.py",
    "tests/test_simulation.py",
    "tests/test_preprocessing.py",
    "tests/test_ekf.py",
    "tests/test_prediction.py",
]

# Keeps otherwise-empty data folders in Git.
GITKEEP_DIRS = ["data/raw", "data/processed", "results", "notebooks"]


def main() -> None:
    for d in DIRECTORIES:
        (ROOT / d).mkdir(parents=True, exist_ok=True)

    created = 0
    for f in PLACEHOLDER_FILES + [f"{d}/.gitkeep" for d in GITKEEP_DIRS]:
        path = ROOT / f
        if not path.exists():
            path.touch()
            created += 1

    print(f"Project skeleton ready at: {ROOT}")
    print(f"Directories ensured: {len(DIRECTORIES)}")
    print(f"New placeholder files created: {created}")


if __name__ == "__main__":
    main()