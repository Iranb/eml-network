import os
import sys
from pathlib import Path


def _looks_like_pytest_invocation() -> bool:
    argv0 = Path(sys.argv[0]).name.lower() if sys.argv else ""
    return argv0.startswith("pytest") or ("-m" in sys.argv and "pytest" in sys.argv)


if _looks_like_pytest_invocation():
    existing = os.environ.get("PYTEST_ADDOPTS", "").strip()
    required = "-p no:capture"
    if required not in existing:
        os.environ["PYTEST_ADDOPTS"] = f"{required} {existing}".strip()
