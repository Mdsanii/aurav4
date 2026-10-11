from hybrid_perception import Box, HybridPerceptionPipeline
from hybrid_perception.stage1_ocr import text_similarity, propose
from hybrid_perception.stage2_verify import verify_candidates
from hybrid_perception.types import Candidate


class FakeImage:
    size = (200, 100)


class FakeOCR:
    def __init__(self, items):
        self.items = items

    def detect(self, image):
        return [Candidate(text=t, box=b, ocr_confidence=c) for t, b, c in self.items]


class FakeVerifier:
    """Scores 1.0 for the box it was told is the real target, 0.0 otherwise."""

    def __init__(self, target, grounded=None):
        self.target = target
        self.grounded = grounded
        self.calls = 0

    def verify(self, image, query, box):
        self.calls += 1
        return (1.0 if box.iou(self.target) > 0.5 else 0.0), self.grounded


def test_iou():
    assert Box(0, 0, 10, 10).iou(Box(0, 0, 10, 10)) == 1.0
    assert Box(0, 0, 10, 10).iou(Box(20, 20, 30, 30)) == 0.0


def test_text_similarity_exact_and_partial():
    assert text_similarity("Sign in", "sign  IN") == 1.0
    assert text_similarity("Sign in", "Sign in to continue") == 0.9
    assert 0 < text_similarity("log in now", "log out") < 0.9
    assert text_similarity("Save", "Cancel") == 0.0


def test_propose_filters_by_match_and_confidence():
    ocr = FakeOCR([
        ("Save", Box(0, 0, 10, 10), 0.99),
        ("Save draft", Box(0, 20, 10, 30), 0.2),   # low confidence -> dropped
        ("Cancel", Box(0, 40, 10, 50), 0.99),      # no match -> dropped
    ])
    out = propose(ocr, FakeImage(), "Save")
    assert [c.text for c in out] == ["Save"]


def test_pipeline_picks_verified_target_over_higher_ocr_match():
    target = Box(100, 10, 150, 30)
    ocr = FakeOCR([
        ("Submit", Box(0, 0, 40, 20), 0.99),   # exact text, but not the real button
        ("Submit", target, 0.90),              # the actual clickable control
    ])
    verifier = FakeVerifier(target)
    pipe = HybridPerceptionPipeline(ocr, verifier)
    res = pipe.locate(FakeImage(), "Submit")
    assert res.found
    assert res.best.box == target
    assert verifier.calls == 2
    assert res.timings.total_ms >= 0


def test_no_proposals_skips_verifier():
    verifier = FakeVerifier(Box(0, 0, 1, 1))
    pipe = HybridPerceptionPipeline(FakeOCR([("Other", Box(0, 0, 5, 5), 0.9)]), verifier)
    res = pipe.locate(FakeImage(), "Submit")
    assert not res.found
    assert verifier.calls == 0


def test_all_rejected_by_verifier_returns_not_found():
    verifier = FakeVerifier(Box(500, 500, 600, 600))  # nothing matches
    pipe = HybridPerceptionPipeline(FakeOCR([("OK", Box(0, 0, 10, 10), 0.9)]), verifier)
    res = pipe.locate(FakeImage(), "OK")
    assert not res.found
    assert res.candidates  # proposals still returned for debugging


def test_verify_candidates_respects_top_k():
    cands = [Candidate("a", Box(0, 0, 1, 1), 0.9, 1.0) for _ in range(10)]
    v = FakeVerifier(Box(0, 0, 1, 1))
    verify_candidates(v, FakeImage(), "a", cands, top_k=3)
    assert v.calls == 3
