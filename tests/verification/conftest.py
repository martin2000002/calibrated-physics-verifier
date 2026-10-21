import sys
from pathlib import Path

# The verification tests reuse the analytic scenes of tests/checks/check_scenarios.py;
# importlib mode does not add that directory to sys.path by itself.
# Golden-case modules next to these tests (oracle_golden_case.py) are imported the same way.
HERE = Path(__file__).resolve().parent
for path in (str(HERE.parent / "checks"), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)
