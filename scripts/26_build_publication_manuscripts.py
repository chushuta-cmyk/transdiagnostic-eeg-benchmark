#!/usr/bin/env python3
"""
Step 26: Entry-Point Publication Manuscript Builder for Paper 2.
Delegates deterministically to Step 32 (`32_build_clean_revised_manuscripts.py` + `32a_manuscript_tables.py`),
ensuring data-driven tables, verified cross-references, sequential [1]..[43] hanging-indented references,
and valid OOXML (`xmlns:m` declared on `<w:settings>`).

Author: Chushu Tan
China University of Political Science and Law
"""

import importlib
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

step32 = importlib.import_module("32_build_clean_revised_manuscripts")


def main():
    step32.build_all()


if __name__ == "__main__":
    main()
