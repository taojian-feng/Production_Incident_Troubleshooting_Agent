import sys
from pathlib import Path

# Make `stacks` importable when running `pytest infra/tests` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
