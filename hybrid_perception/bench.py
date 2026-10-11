"""Accuracy and latency harness.

Labels file (JSONL), one example per line:
  {"image": "path/to.png", "query": "Sign in", "box": [x1, y1, x2, y2]}

Accuracy = fraction of examples whose top-1 box has IoU >= iou_thr with the label.
Latency  = median and p90 of end-to-end locate() time.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .types import Box


def percentile(values, p):
    s = sorted(values)
    if not s:
        return float("nan")
    k = min(len(s) - 1, max(0, int(round(p / 100 * (len(s) - 1)))))
    return s[k]


def run(pipeline, labels_path: Path, iou_thr: float = 0.5):
    from PIL import Image

    rows = [json.loads(l) for l in labels_path.read_text().splitlines() if l.strip()]
    hits, latencies, misses = 0, [], []
    for row in rows:
        image = Image.open(row["image"]).convert("RGB")
        result = pipeline.locate(image, row["query"])
        latencies.append(result.timings.total_ms)
        gt = Box(*row["box"])
        ok = result.found and result.best.box.iou(gt) >= iou_thr
        hits += int(ok)
        if not ok:
            misses.append({"query": row["query"], "image": row["image"],
                           "found": result.found})
    n = len(rows)
    return {
        "n": n,
        "accuracy": hits / n if n else float("nan"),
        "latency_median_ms": statistics.median(latencies) if latencies else float("nan"),
        "latency_p90_ms": percentile(latencies, 90),
        "misses": misses,
    }


def main(argv=None):
    from .pipeline import HybridPerceptionPipeline
    from .stage1_ocr import RapidOCRBackend
    from .stage2_verify import Florence2Verifier

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", type=Path)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--florence-model", default=Florence2Verifier.MODEL_ID)
    ap.add_argument("--out", type=Path, default=Path("bench_results/last.json"))
    args = ap.parse_args(argv)

    pipe = HybridPerceptionPipeline(RapidOCRBackend(), Florence2Verifier(args.florence_model))
    report = run(pipe, args.labels, iou_thr=args.iou)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "misses"}, indent=2))


if __name__ == "__main__":
    main()
