from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineDefaults:
    sample_every_n: int = 30
    top_n_sharp: int = 150
    gemini_selects: int = 10
    thumb_w: int = 320
    thumb_h: int = 240
    grid_cols: int = 2
    gemini_model: str = "gemini-2.5-flash"
    gemini_batch: int = 30
    temporal_gap: int = 90
    max_retries: int = 3
    retry_delay: float = 2.0


@dataclass(frozen=True)
class Theme:
    bg: str = "#0b0d10"
    panel: str = "#12161b"
    panel_alt: str = "#171c22"
    border: str = "#28303a"
    border_strong: str = "#3b4654"
    text: str = "#edf2f7"
    text_muted: str = "#94a3b8"
    text_soft: str = "#64748b"
    accent: str = "#ff8a3d"
    accent_soft: str = "#ffd3bd"
    success: str = "#78e08f"
    danger: str = "#ff6b6b"


HOW_IT_WORKS = (
    "OpenCV samples the video, scores each frame for sharpness, and prunes near-duplicates. "
    "Gemini then curates the strongest reactions so the final grid feels expressive, varied, and printable."
)

