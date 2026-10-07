"""Verify that the environment and project structure are ready."""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "plotly": "plotly",
    "streamlit": "streamlit",
    "yaml": "PyYAML",
    "pytest": "pytest",
}

REQUIRED_PATHS = [
    "config/config.yaml",
    "data/raw",
    "data/processed",
    "src/__init__.py",
    "src/simulation/__init__.py",
    "src/preprocessing/__init__.py",
    "src/ekf/__init__.py",
    "src/prediction/__init__.py",
    "src/pipeline.py",
    "dashboard/app.py",
    "tests/test_ekf.py",
]


def check_python() -> bool:
    ok = sys.version_info >= (3, 10)
    v = sys.version_info
    print(f"[{'OK' if ok else 'FAIL'}] Python {v.major}.{v.minor}.{v.micro} (need >= 3.10)")
    return ok


def check_packages() -> bool:
    all_ok = True
    for module, pip_name in REQUIRED_PACKAGES.items():
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "unknown")
            print(f"[OK]   {pip_name:<10} {version}")
        except ImportError:
            print(f"[FAIL] {pip_name} is not installed -> pip install -r requirements.txt")
            all_ok = False
    return all_ok


def check_structure() -> bool:
    all_ok = True
    for rel in REQUIRED_PATHS:
        exists = (ROOT / rel).exists()
        if not exists:
            print(f"[FAIL] Missing: {rel}")
            all_ok = False
    if all_ok:
        print(f"[OK]   All {len(REQUIRED_PATHS)} required paths exist")
    return all_ok


def check_package_import() -> bool:
    sys.path.insert(0, str(ROOT))
    try:
        importlib.import_module("src")
        importlib.import_module("src.ekf")
        importlib.import_module("src.simulation")
        print("[OK]   'src' is importable as a Python package")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] Cannot import src: {exc}")
        return False


if __name__ == "__main__":
    print("=== Smart IV Infusion Monitor: setup check ===")
    results = [check_python(), check_packages(), check_structure(), check_package_import()]
    print("==============================================")
    if all(results):
        print("ALL CHECKS PASSED. Ready for Step 2.")
        sys.exit(0)
    print("Some checks failed. Fix the items above and re-run.")
    sys.exit(1)