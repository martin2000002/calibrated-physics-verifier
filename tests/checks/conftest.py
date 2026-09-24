import sys
from pathlib import Path

# check_scenarios.py lives next to the tests; importlib mode does not add this directory to
# sys.path by itself.
HERE = str(Path(__file__).resolve().parent)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
