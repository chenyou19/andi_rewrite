"""Export participant-level rosters from a frozen model-grid v3 revision.

This is a read-only report utility.  It reads the explicit frozen
``manifests/<comparison>/<split>.jsonl`` files and writes new CSV/audit files
under the requested output directory; it never edits the build revision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from domain_classifier.report import export_model_grid_v3_rosters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_root", type=Path, help="Frozen model_grid_v3 revision containing manifests/")
    parser.add_argument("output_dir", type=Path, help="New directory for rosters and overlap audit")
    args = parser.parse_args()
    result = export_model_grid_v3_rosters(args.build_root, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
