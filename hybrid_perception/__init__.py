"""Hybrid perception pipeline: OCR proposals verified by a visual grounding model."""

from .types import Box, Candidate, PipelineResult, Timings
from .pipeline import HybridPerceptionPipeline

__all__ = ["Box", "Candidate", "PipelineResult", "Timings", "HybridPerceptionPipeline"]
