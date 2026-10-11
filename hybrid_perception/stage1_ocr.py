"""Stage 1: text and layout extraction with RapidOCR (PaddleOCR-compatible output)."""

from __future__ import annotations

from typing import List, Protocol

from .types import Box, Candidate


class OCRBackend(Protocol):
    def detect(self, image) -> List[Candidate]:  # image: PIL.Image or ndarray
        ...


class RapidOCRBackend:
    """Wraps rapidocr_onnxruntime. Imported lazily so the package works without it."""

    def __init__(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "RapidOCR not installed; `pip install .[ocr]`"
            ) from exc
        self._engine = RapidOCR()

    def detect(self, image) -> List[Candidate]:
        import numpy as np

        arr = np.asarray(image.convert("RGB")) if hasattr(image, "convert") else image
        result, _ = self._engine(arr)
        out: List[Candidate] = []
        for poly, text, conf in result or []:
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            out.append(
                Candidate(
                    text=str(text),
                    box=Box(min(xs), min(ys), max(xs), max(ys)),
                    ocr_confidence=float(conf),
                )
            )
        return out


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def text_similarity(query: str, text: str) -> float:
    """Token-overlap F1 with a substring bonus. Cheap, deterministic, no extra deps."""
    q, t = normalize(query), normalize(text)
    if not q or not t:
        return 0.0
    if q == t:
        return 1.0
    if q in t or t in q:
        return 0.9
    qs, ts = set(q.split()), set(t.split())
    common = len(qs & ts)
    if common == 0:
        return 0.0
    precision, recall = common / len(ts), common / len(qs)
    return 2 * precision * recall / (precision + recall)


def propose(backend: OCRBackend, image, query: str, min_match: float = 0.5,
            min_conf: float = 0.5) -> List[Candidate]:
    """Run OCR and keep regions whose text plausibly matches the query."""
    candidates = [c for c in backend.detect(image) if c.ocr_confidence >= min_conf]
    for c in candidates:
        c.text_match = text_similarity(query, c.text)
    kept = [c for c in candidates if c.text_match >= min_match]
    kept.sort(key=lambda c: (c.text_match, c.ocr_confidence), reverse=True)
    return kept
