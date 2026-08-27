from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from time_causality_eval.scripts.tc_common import load_yaml
from time_causality_eval.scripts.time_causality_pipeline import cluster_events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "time_causality_eval" / "configs" / "time_causality_config.yaml"))
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out = cluster_events(cfg, args.project_root, args.output_root, skip_existing=args.skip_existing)
    print(out)


if __name__ == "__main__":
    main()
