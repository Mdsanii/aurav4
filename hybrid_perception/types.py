from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in absolute pixel coordinates (x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: "Box") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def pad(self, px: float) -> "Box":
        return Box(self.x1 - px, self.y1 - px, self.x2 + px, self.y2 + px)

    def clip(self, width: int, height: int) -> "Box":
        return Box(
            max(0.0, self.x1),
            max(0.0, self.y1),
            min(float(width), self.x2),
            min(float(height), self.y2),
        )


@dataclass
class Candidate:
    """A text region proposed by Stage 1 (OCR) as a possible click target."""

    text: str
    box: Box
    ocr_confidence: float
    text_match: float = 0.0       # 0..1 similarity between text and the query
    verify_score: Optional[float] = None  # filled by Stage 2
    verified_box: Optional[Box] = None    # grounding model's box, if any


@dataclass
class Timings:
    stage1_ms: float = 0.0
    stage2_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.stage1_ms + self.stage2_ms


@dataclass
class PipelineResult:
    query: str
    best: Optional[Candidate]
    candidates: list = field(default_factory=list)
    timings: Timings = field(default_factory=Timings)

    @property
    def found(self) -> bool:
        return self.best is not None
