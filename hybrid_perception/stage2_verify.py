"""Stage 2: visual grounding verification of Stage 1 candidates.

A verifier receives the full image, the query, and one candidate box. It returns
a score in [0, 1] for whether that box is the query's clickable target, plus the
grounding model's own box (if it produced one) for refinement.
"""

from __future__ import annotations

from typing import Optional, Protocol, Tuple

from .types import Box


class Verifier(Protocol):
    def verify(self, image, query: str, box: Box) -> Tuple[float, Optional[Box]]:
        ...


class Florence2Verifier:
    """Florence-2 phrase grounding on a padded crop around the candidate.

    Lazy-loads microsoft/Florence-2-large (trust_remote_code). The crop keeps the
    model's input focused on the candidate; the returned box is mapped back to
    full-image coordinates and scored by IoU against the OCR box.
    """

    MODEL_ID = "microsoft/Florence-2-large"

    def __init__(self, model_id: str = MODEL_ID, device: Optional[str] = None,
                 pad_px: int = 24, task: str = "<CAPTION_TO_PHRASE_GROUNDING>") -> None:
        import torch  # optional extra: florence
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.dtype = dtype
        self.pad_px = pad_px
        self.task = task

    def verify(self, image, query: str, box: Box) -> Tuple[float, Optional[Box]]:
        from PIL import Image  # noqa: F401  (image is a PIL.Image)

        W, H = image.size
        region = box.pad(self.pad_px).clip(W, H)
        crop = image.crop((int(region.x1), int(region.y1), int(region.x2), int(region.y2)))
        prompt = self.task + query
        inputs = self.processor(text=prompt, images=crop, return_tensors="pt").to(
            self.device, self.dtype
        )
        with self._torch.no_grad():
            ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=128,
                num_beams=1,
                do_sample=False,
            )
        raw = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            raw, task=self.task, image_size=crop.size
        )
        boxes = parsed[self.task].get("bboxes", [])
        if not boxes:
            return 0.0, None
        bx1, by1, bx2, by2 = boxes[0]
        # Map crop-local coordinates back to the full image.
        mapped = Box(bx1 + region.x1, by1 + region.y1, bx2 + region.x1, by2 + region.y1)
        return mapped.iou(box), mapped


def verify_candidates(verifier: Verifier, image, query: str, candidates, top_k: int = 5,
                      min_score: float = 0.1):
    """Verify the top-k candidates and return them sorted by verify_score."""
    W, H = image.size
    scored = []
    for cand in candidates[:top_k]:
        score, grounded = verifier.verify(image, query, cand.box.clip(W, H))
        cand.verify_score = score
        cand.verified_box = grounded
        if score >= min_score:
            scored.append(cand)
    scored.sort(key=lambda c: (c.verify_score, c.text_match, c.ocr_confidence), reverse=True)
    return scored
