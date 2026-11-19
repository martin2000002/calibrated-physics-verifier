import sys
from pathlib import Path

# The verification tests reuse the analytic scenes of tests/checks/check_scenarios.py and the
# stage-key and error-profile builders of tests/extraction/support.py; importlib mode does not
# add those directories to sys.path by itself.
# Golden-case modules next to these tests (oracle_golden_case.py) are imported the same way.
HERE = Path(__file__).resolve().parent
for path in (str(HERE.parent / "checks"), str(HERE.parent / "extraction"), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)
