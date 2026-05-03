"""Stacked bar chart of per-preset runtime split into DD and NST.

Style is kept consistent with make_runtime_chart.py (same colors, fonts, dpi).

Usage:
    python analysis/make_preset_timing_chart.py [report.json] [out.png]
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "analysis" / "experiments" / "runtime_report_preset_timing.json"
DEFAULT_OUT    = ROOT / "Thesis" / "DeepDream_NST_szakdolgozat" / "Pictures" / "preset_timing_chart.png"

# Stable ordering; filtered to what's present in the report
PRESET_ORDER = ["GENTLE", "FAST", "Default", "PORTRAIT", "PAINTERLY", "LSD", "DEEP", "HEAVY"]

# Match runtime_chart.png palette
COLOR_DD  = "#7c6aff"
COLOR_NST = "#ff6a8a"


def main():
    report = Path(sys.argv[1]) if len(sys.argv) >= 2 else DEFAULT_REPORT
    out    = Path(sys.argv[2]) if len(sys.argv) >= 3 else DEFAULT_OUT

    if not report.exists():
        print(f"No report at {report}")
        sys.exit(1)

    data = json.loads(report.read_text())
    by_preset = {r["preset"]: r for r in data if r.get("category") == "preset_timing"}

    presets = [p for p in PRESET_ORDER if p in by_preset]
    dd      = [by_preset[p].get("dd_s", 0.0)  for p in presets]
    nst     = [by_preset[p].get("nst_s", 0.0) for p in presets]
    total   = [by_preset[p].get("runtime_s", 0.0) for p in presets]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(presets))

    ax.bar(x, dd,  color=COLOR_DD,  label="DeepDream")
    ax.bar(x, nst, bottom=dd, color=COLOR_NST, label="Neural Style Transfer")

    for i, t in enumerate(total):
        ax.text(i, t + 0.3, f"{t:.1f}s", ha="center", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(presets, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Futási idő (s)")
    ax.set_title("Presetenkénti futási idő — DeepDream vs NST szétbontásban")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Futási idők — thesis_configs/experiments_preset_timing.json", fontsize=13)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved: {out}")

    total_all = sum(total)
    nst_all   = sum(nst)
    if total_all > 0:
        print(f"NST részesedés összesítve: {nst_all / total_all * 100:.1f}%  "
              f"(NST={nst_all:.1f}s, össz={total_all:.1f}s)")


if __name__ == "__main__":
    main()
