"""Render a metrics diagram from an adaptive pipeline experiment_log.json.

Reuses `_plot_metrics` from backend.app.services.orchestrator.

Usage:
    python analysis/plot_experiment_log.py <log.json> [out.png]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.services.orchestrator import _plot_metrics


def main():
    if len(sys.argv) < 2:
        print("Usage: plot_experiment_log.py <log.json> [out.png]")
        sys.exit(1)

    log_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else log_path.with_name("metrics_plot.png")

    with open(log_path, "r") as f:
        log = json.load(f)

    _plot_metrics(log, str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
