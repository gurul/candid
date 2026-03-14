from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    frames: int
    duration_s: float
    width: int
    height: int


@dataclass(frozen=True)
class TranscriptChunk:
    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True)
class QuestionWindow:
    question: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class SelectedFrame:
    frame_bgr: Any
    score: float
    frame_index: int
    question_label: Optional[str] = None


@dataclass(frozen=True)
class ExtractionOptions:
    sample_every: int
    top_n: int
    min_gap: int
    n_select: int
    gemini_key: str
    openai_key: str = ""
    questions: tuple[str, ...] = field(default_factory=tuple)

    @property
    def use_question_mode(self) -> bool:
        return bool(self.openai_key and len(self.questions) >= 4 and all(self.questions[:4]))

