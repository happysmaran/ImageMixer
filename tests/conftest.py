import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# The suite must not depend on the network: the trained cutout model is downloaded on first use, so it is
# switched off here and exercised by a dedicated test that points at an already-cached copy.
os.environ.setdefault("FLATLAY_NO_MODEL", "1")
