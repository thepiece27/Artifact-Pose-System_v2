"""Make ``app`` and the test helpers importable however pytest is launched."""
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
for p in (TESTS.parent, TESTS, TESTS / "cita"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
