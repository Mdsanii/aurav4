# Hybrid Perception Pipeline

Locates clickable UI targets by splitting the work in two:

1. **Stage 1: text and layout extraction.** RapidOCR (`rapidocr-onnxruntime`) returns text boxes. Boxes whose text matches the query are kept as candidates.
2. **Stage 2: visual grounding verification.** Florence-2-Large (`microsoft/Florence-2-large`) grounds the query inside a padded crop around each of the top-k candidates. A candidate is kept when the grounded box agrees with the OCR box.

The pipeline returns the best verified candidate, or `found=False` so the caller can fall back to a full VLM pass.

## Layout

```
hybrid_perception/
  types.py          Box, Candidate, Timings, PipelineResult
  stage1_ocr.py     RapidOCR backend, text matching, proposal filtering
  stage2_verify.py  Florence-2 verifier, candidate verification
  pipeline.py       HybridPerceptionPipeline.locate()
  bench.py          accuracy (IoU hit@1) and latency (median, p90) harness
tests/              unit tests with fake OCR and verifier (no models needed)
```

## Setup

```bash
pip install -e ".[dev,ocr]"          # tests + RapidOCR
pip install -e ".[florence]"         # Stage 2 model (torch, transformers)
pytest
```

## Benchmark

Create a JSONL label file (`image`, `query`, `box` = [x1, y1, x2, y2] in pixels), then:

```bash
python -m hybrid_perception.bench labels.jsonl --iou 0.5
```

## Status and targets

The targets from the proposal (>92% accuracy, median latency < 1.2 s) are **not yet measured**. They must be checked with `bench.py` on a labeled set of real UI screenshots. Florence-2 CPU latency in particular is unknown and may exceed the budget; this is the first thing to measure.

The Florence-2 verifier has not been run against the real model in this environment. Its output parsing follows the Florence-2 remote-code API and should be verified on first run.
