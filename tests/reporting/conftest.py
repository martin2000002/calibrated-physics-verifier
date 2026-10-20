import sys
from pathlib import Path

# The results tests build their artifact with the calibration builders and share
# reporting_cases.py; importlib mode does not add these directories to sys.path by itself.
HERE = Path(__file__).resolve().parent
for folder in (HERE, HERE.parent / "calibration"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
