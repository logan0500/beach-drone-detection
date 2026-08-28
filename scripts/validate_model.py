#!/usr/bin/env python3
"""Validate a trained model against a labeled test set: precision, recall, mAP.

This answers "does the model actually find what it's supposed to find" -
the numbers to check before trusting a detector or telling anyone it works,
especially swimmer_distress which hasn't been validated yet.

You need a *labeled* test set for this (images + ground-truth boxes), not
just a folder of images - a folder of images alone has nothing to compare
predictions against. The easiest way to get one: in Roboflow, on the same
project you trained from, use a "test" split that was held out of training
(or generate one), and export it in "YOLOv8" format. That gives you a
data.yaml plus images/labels folders - point this script at that data.yaml.

Usage:
    python scripts/validate_model.py --list
    python scripts/validate_model.py shark --data path/to/test_set/data.yaml
    python scripts/validate_model.py swimmer_distress --data path/to/test_set/data.yaml --report report.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beach_drone_detection.pipeline.config import load_config

ALIASES = {
    "shark": "shark-best",
    "marine_life": "shark-best",
    "rip": "rip-best",
    "rip_current": "rip-best",
    "swimmer_multiclass": "swimmer-multiclass-best",
    "isolated_swimmer": "swimmer-multiclass-best",
    "vessel_encroachment": "swimmer-multiclass-best",
    "swimmer_distress": "swimmer-distress-best",
    "distress": "swimmer-distress-best",
}


def discover_models(config: dict) -> dict:
    """Model file stem -> {"path", "conf_threshold"}, deduped across detectors
    that share the same underlying model file (e.g. swimmer-multiclass-best.pt
    backs both isolated_swimmer and vessel_encroachment)."""
    models = {}
    for _, cfg in config.get("detectors", {}).items():
        path = cfg.get("model_path")
        if not path:
            continue
        stem = Path(path).stem
        models.setdefault(stem, {"path": path, "conf_threshold": cfg.get("conf_threshold", 0.25)})
    return models


def print_table(rows, overall):
    headers = ["Class", "Images", "Instances", "Precision", "Recall", "mAP50", "mAP50-95"]
    widths = [18, 8, 10, 10, 8, 8, 9]
    print("".join(h.ljust(w) for h, w in zip(headers, widths)))
    for row in rows:
        values = [
            row["Class"], row["Images"], row["Instances"],
            row["Box-P"], row["Box-R"], row["mAP50"], row["mAP50-95"],
        ]
        print("".join(str(v).ljust(w) for v, w in zip(values, widths)))
    print("-" * sum(widths))
    values = ["ALL", "", "", overall["precision"], overall["recall"], overall["mAP50"], overall["mAP50_95"]]
    print("".join(str(v).ljust(w) for v, w in zip(values, widths)))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "model", nargs="?",
        help="Model name: shark, rip_current, swimmer_multiclass, swimmer_distress (aliases OK)",
    )
    parser.add_argument("--data", help="Path to the test set's data.yaml (Roboflow YOLOv8 export)")
    parser.add_argument("--conf", type=float, default=None, help="Override the scoring confidence threshold")
    parser.add_argument("--config", default="configs/default.yaml", help="Pipeline config to read model paths/thresholds from")
    parser.add_argument("--list", action="store_true", help="List available model names and exit")
    parser.add_argument("--report", default=None, help="Optional path to save a JSON report")
    args = parser.parse_args()

    config = load_config(args.config)
    models = discover_models(config)

    if args.list or not args.model:
        print("Available models:")
        for stem, info in models.items():
            print(f"  {stem}  ({info['path']})")
        return

    if not args.data:
        parser.error("--data is required: path to the test set's data.yaml")

    stem = ALIASES.get(args.model, args.model)
    if stem not in models:
        parser.error(f"Unknown model '{args.model}'. Run with --list to see options.")

    info = models[stem]
    conf = args.conf if args.conf is not None else info["conf_threshold"]

    from ultralytics import YOLO

    model = YOLO(info["path"])
    print(f"Validating {info['path']} against {args.data} (conf={conf})...\n")
    metrics = model.val(data=args.data, conf=conf, split="test", plots=False, save_json=False)

    rows = metrics.summary()
    overall = {
        "precision": round(float(metrics.box.mp), 3),
        "recall": round(float(metrics.box.mr), 3),
        "mAP50": round(float(metrics.box.map50), 3),
        "mAP50_95": round(float(metrics.box.map), 3),
    }

    print()
    print_table(rows, overall)

    if overall["recall"] < 0.5:
        print(
            f"\nRecall is {overall['recall']:.0%} - this model is missing more than half of "
            f"what it should be finding on this test set. Not ready to rely on."
        )

    if args.report:
        report = {
            "model": info["path"],
            "data": args.data,
            "conf": conf,
            "per_class": rows,
            "overall": overall,
        }
        Path(args.report).write_text(json.dumps(report, indent=2, default=str))
        print(f"\nSaved report to {args.report}")


if __name__ == "__main__":
    main()
