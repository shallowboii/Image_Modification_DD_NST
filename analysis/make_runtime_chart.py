"""Build a runtime bar chart from runtime_report.json.

Groups runs by category (or preset) and plots average wall-clock time in seconds.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "analysis" / "experiments" / "runtime_report.json"
OUT    = ROOT / "Thesis" / "DeepDream_NST_szakdolgozat" / "Pictures" / "runtime_chart.png"


def main():
    if not REPORT.exists():
        print(f"No report at {REPORT}")
        sys.exit(1)

    data = json.loads(REPORT.read_text())

    # Organize: preset -> list of (label, runtime)
    by_preset = defaultdict(list)
    for r in data:
        if "runtime_s" not in r:
            continue
        label = f"{Path(r['input']).stem} [{r.get('mode','fixed')}]"
        by_preset[r["preset"]].append((label, r["runtime_s"]))

    presets = list(by_preset.keys())
    fig, axes = plt.subplots(1, len(presets), figsize=(5 * len(presets), 5), squeeze=False)
    axes = axes[0]

    colors = ["#7c6aff", "#ff6a8a"]
    for ax, preset in zip(axes, presets):
        entries = by_preset[preset]
        labels, times = zip(*entries)
        bars = ax.bar(range(len(entries)), times,
                      color=[colors[i % len(colors)] for i in range(len(entries))])
        ax.set_xticks(range(len(entries)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Futási idő (s)")
        ax.set_title(f"Preset: {preset}")
        ax.grid(axis="y", alpha=0.3)
        for b, t in zip(bars, times):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                    f"{t:.1f}s", ha="center", fontsize=9)

    fig.suptitle("Futási idők — thesis_configs/experiments_config.json", fontsize=13)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=120)
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
