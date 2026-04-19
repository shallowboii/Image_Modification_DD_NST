"""Reproducible experiment runner for the thesis validation matrix.

Reads a JSON config describing (input, style, preset, mode) combinations,
runs each through the orchestrator, saves the result to a structured folder
tree, and records runtime in a single report file.

Usage:
    python analysis/run_experiments.py analysis/experiments_config.json

Output layout:
    analysis/experiments/
        preset_comparison/<input_stem>/<preset>.jpg
        multi_input/<preset>/<input_stem>.jpg
        multi_style/<input_stem>/<style_stem>.jpg
        adaptive_vs_fixed/<preset>/<input_stem>__<mode>.jpg
        runtime_report.json
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from backend.app.services.orchestrator import (
    run_cyclic_pipeline,
    run_adaptive_cyclic_pipeline,
)

EXPERIMENTS_DIR = ROOT / "analysis" / "experiments"
REPORT_PATH     = EXPERIMENTS_DIR / "runtime_report.json"


def _rel(p):
    return str(Path(p).relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p)


def _target_path(task):
    category = task["category"]
    stem_in  = Path(task["input"]).stem
    preset   = task.get("preset", "Default")
    mode     = task.get("mode", "fixed")
    style    = task.get("style")
    stem_sty = Path(style).stem if style else "default"

    if category == "preset_comparison":
        return EXPERIMENTS_DIR / category / stem_in / f"{preset}.jpg"
    if category == "multi_input":
        return EXPERIMENTS_DIR / category / preset / f"{stem_in}.jpg"
    if category == "multi_style":
        return EXPERIMENTS_DIR / category / stem_in / f"{stem_sty}.jpg"
    if category == "adaptive_vs_fixed":
        return EXPERIMENTS_DIR / category / preset / f"{stem_in}__{mode}.jpg"
    return EXPERIMENTS_DIR / category / f"{stem_in}_{preset}_{mode}.jpg"


def _run_one(task):
    input_path = str(ROOT / task["input"])
    preset     = task.get("preset", "Default")
    mode       = task.get("mode", "fixed")
    style      = task.get("style")

    params = {"preset": preset}
    if style:
        params["style"] = style

    dump_dir = EXPERIMENTS_DIR / "_tmp" / f"{int(time.time()*1000)}_{Path(input_path).stem}_{preset}_{mode}"
    dump_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if mode == "adaptive":
        out = run_adaptive_cyclic_pipeline(input_path, params, str(dump_dir), plot_metrics=True)
    else:
        out = run_cyclic_pipeline(input_path, params, str(dump_dir))
    elapsed = time.perf_counter() - t0

    target = _target_path(task)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out, target)

    log_target = None
    if mode == "adaptive":
        src_log = Path(dump_dir) / "experiment_log.json"
        if src_log.exists():
            log_target = target.with_suffix(".log.json")
            shutil.copy2(src_log, log_target)

    return {
        "category":  task["category"],
        "input":     task["input"],
        "style":     style,
        "preset":    preset,
        "mode":      mode,
        "output":    _rel(target),
        "log":       _rel(log_target) if log_target else None,
        "runtime_s": round(elapsed, 2),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: run_experiments.py <config.json>")
        sys.exit(1)

    cfg_path = Path(sys.argv[1])
    tasks = json.loads(cfg_path.read_text())

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    print(f"[runner] {len(tasks)} task(s) to execute")
    for i, task in enumerate(tasks, 1):
        label = f"{task['category']} | {task.get('preset','-')} | {Path(task['input']).stem} | {task.get('mode','fixed')}"
        print(f"\n[runner] ({i}/{len(tasks)}) {label}")
        try:
            rec = _run_one(task)
            results.append(rec)
            print(f"  OK  -> {rec['output']}  ({rec['runtime_s']} s)")
        except Exception as e:
            print(f"  FAIL: {e}")
            results.append({**task, "error": str(e)})

        REPORT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\n[runner] Report: {_rel(REPORT_PATH)}")


if __name__ == "__main__":
    main()
