from __future__ import annotations

import time
from typing import Optional

from . import stage1_ocr, stage2_verify
from .types import Candidate, PipelineResult, Timings


class HybridPerceptionPipeline:
    """Stage 1 (OCR proposals) -> Stage 2 (grounding verification) -> best clickable target.

    If Stage 1 finds nothing, Stage 2 is skipped and the result is `found == False`.
    Callers that need a fallback (e.g. a full VLM pass) should check `result.found`.
    """

    def __init__(self, ocr_backend, verifier, *, min_text_match: float = 0.5,
                 min_ocr_conf: float = 0.5, top_k: int = 5, min_verify: float = 0.1):
        self.ocr = ocr_backend
        self.verifier = verifier
        self.min_text_match = min_text_match
        self.min_ocr_conf = min_ocr_conf
        self.top_k = top_k
        self.min_verify = min_verify

    def locate(self, image, query: str) -> PipelineResult:
        t0 = time.perf_counter()
        proposals = stage1_ocr.propose(
            self.ocr, image, query, min_match=self.min_text_match, min_conf=self.min_ocr_conf
        )
        t1 = time.perf_counter()
        verified = []
        if proposals:
            verified = stage2_verify.verify_candidates(
                self.verifier, image, query, proposals,
                top_k=self.top_k, min_score=self.min_verify,
            )
        t2 = time.perf_counter()

        best: Optional[Candidate] = verified[0] if verified else None
        return PipelineResult(
            query=query,
            best=best,
            candidates=verified or proposals,
            timings=Timings(stage1_ms=(t1 - t0) * 1000, stage2_ms=(t2 - t1) * 1000),
        )
