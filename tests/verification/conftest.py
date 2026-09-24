import sys
from pathlib import Path

# The verification tests reuse the analytic scenes of tests/checks/check_scenarios.py;
# importlib mode does not add that directory to sys.path by itself.
CHECKS_TESTS = str(Path(__file__).resolve().parent.parent / "checks")
if CHECKS_TESTS not in sys.path:
    sys.path.insert(0, CHECKS_TESTS)
