import io
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from typing import Callable, Optional

import cv2
import google.genai as genai
import numpy as np
from PIL import Image

from .config import PipelineDefaults
from .models import ExtractionOptions, QuestionWindow, SelectedFrame, TranscriptChunk, VideoMetadata


ProgressValue = Optional[float]
StatusCallback = Optional[Callable[[Optional[str], ProgressValue], None]]

DEFAULTS = PipelineDefaults()


def _emit(progress_cb: StatusCallback, message: Optional[str], value: ProgressValue) -> None:
    if progress_cb:
        progress_cb(message, value)


def _cancelled(cancel_event: Optional[threading.Event]) -> bool:
    return bool(cancel_event and cancel_event.is_set())


def get_video_metadata(video_path: str) -> VideoMetadata:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 1
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return VideoMetadata(
        fps=fps,
        frames=frames,
        duration_s=frames / fps if fps else 0,
        width=width,
        height=height,
    )


def extract_sharp_frames(
    video_path: str,
    sample_every: int,
    cancel_event: Optional[threading.Event] = None,
    frame_progress: Optional[Callable[[int, int], None]] = None,
) -> list[SelectedFrame]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[SelectedFrame] = []
    idx = 0
    while True:
        if _cancelled(cancel_event):
            cap.release()
            return []
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            frames.append(SelectedFrame(frame_bgr=frame.copy(), score=score, frame_index=idx))
        idx += 1
        if frame_progress:
            frame_progress(idx, max(total, 1))
    cap.release()
    return sorted(frames, key=lambda item: item.score, reverse=True)


def extract_sharp_frames_in_range(
    video_path: str,
    start_s: float,
    end_s: float,
    sample_every: int,
    cancel_event: Optional[threading.Event] = None,
) -> list[SelectedFrame]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_idx = max(0, int(start_s * fps))
    end_idx = min(total_frames - 1, int(end_s * fps))
    if start_idx >= end_idx:
        cap.release()
        return []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    frames: list[SelectedFrame] = []
    idx = start_idx
    while idx <= end_idx:
        if _cancelled(cancel_event):
            cap.release()
            return []
        ret, frame = cap.read()
        if not ret:
            break
        if (idx - start_idx) % sample_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            frames.append(SelectedFrame(frame_bgr=frame.copy(), score=score, frame_index=idx))
        idx += 1
    cap.release()
    return sorted(frames, key=lambda item: item.score, reverse=True)


def temporal_diversity_filter(frames: list[SelectedFrame], min_gap: int, limit: int) -> list[SelectedFrame]:
    if min_gap <= 0:
        return frames[:limit]
    kept: list[SelectedFrame] = []
    last_frame = -min_gap - 1
    for item in frames:
        if item.frame_index - last_frame >= min_gap:
            kept.append(item)
            last_frame = item.frame_index
            if len(kept) >= limit:
                break
    return kept


def _frame_part(frame_bgr) -> genai.types.Part:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    img.thumbnail((DEFAULTS.thumb_w, DEFAULTS.thumb_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return genai.types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


def _extract_json_block(text: str, key: str) -> Optional[dict]:
    match = re.search(r"\{[^{}]*\"" + re.escape(key) + r"\"[^{}]*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def _retryable_gemini_call(client: genai.Client, contents, cancel_event: Optional[threading.Event]) -> str:
    for attempt in range(DEFAULTS.max_retries):
        if _cancelled(cancel_event):
            return ""
        try:
            response = client.models.generate_content(model=DEFAULTS.gemini_model, contents=contents)
            return (response.text or "").strip()
        except Exception:
            if attempt >= DEFAULTS.max_retries - 1:
                raise
            time.sleep(DEFAULTS.retry_delay * (attempt + 1))
    return ""


def _gemini_pick_batch(
    client: genai.Client,
    batch_frames: list[SelectedFrame],
    total_select: int,
    cancel_event: Optional[threading.Event],
) -> list[int]:
    prompt = (
        "You are a professional photo editor curating video frames.\n"
        "Select expressive, flattering, sharp, varied frames with good eye contact, clean crops, and natural skin tone.\n"
        "Avoid blinks, heavy motion blur, awkward mid-speech mouth shapes, and near-duplicates.\n"
        f"There are {len(batch_frames)} frames indexed 0 to {len(batch_frames) - 1}.\n"
        f"Return exactly {min(total_select, len(batch_frames))} indices in quality order.\n"
        'Return only JSON: {"indices": [0, 1, 2]}'
    )
    contents = [prompt]
    for idx, frame in enumerate(batch_frames):
        contents.append(f"\nFrame {idx}:")
        contents.append(_frame_part(frame.frame_bgr))

    text = _retryable_gemini_call(client, contents, cancel_event)
    payload = _extract_json_block(text, "indices")
    if payload and isinstance(payload.get("indices"), list):
        return [i for i in payload["indices"] if isinstance(i, int) and 0 <= i < len(batch_frames)]
    return [i for i in map(int, re.findall(r"\b\d+\b", text)) if 0 <= i < len(batch_frames)]


def gemini_select(
    top_frames: list[SelectedFrame],
    api_key: str,
    n_select: int,
    progress_cb: StatusCallback = None,
    cancel_event: Optional[threading.Event] = None,
) -> list[SelectedFrame]:
    if not top_frames:
        return []
    client = genai.Client(api_key=api_key)
    batches = [top_frames[i:i + DEFAULTS.gemini_batch] for i in range(0, len(top_frames), DEFAULTS.gemini_batch)]
    per_batch = max(3, n_select // max(len(batches), 1) + 1)
    finalists: list[SelectedFrame] = []

    for idx, batch in enumerate(batches, start=1):
        if _cancelled(cancel_event):
            return []
        _emit(progress_cb, f"Gemini curation batch {idx}/{len(batches)}...", 52 + (idx - 1) / max(len(batches), 1) * 22)
        for local_idx in _gemini_pick_batch(client, batch, per_batch, cancel_event):
            finalists.append(batch[local_idx])

    if len(finalists) <= n_select:
        return finalists

    _emit(progress_cb, "Finalizing shortlist...", 80)
    final_idx = _gemini_pick_batch(client, finalists, n_select, cancel_event)
    return [finalists[i] for i in final_idx[:n_select]]


def transcribe_video_to_chunks(
    video_path: str,
    openai_api_key: str,
    cancel_event: Optional[threading.Event] = None,
) -> list[TranscriptChunk]:
    if _cancelled(cancel_event):
        return []
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The OpenAI package is required for photo-strip mode. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        audio_path = tmp.name

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "libmp3lame", "-q:a", "4",
                "-loglevel", "error", audio_path,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr or result.stdout}")
        if _cancelled(cancel_event):
            return []

        client = OpenAI(api_key=openai_api_key)
        with open(audio_path, "rb") as handle:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=handle,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        chunks = getattr(response, "segments", None) or []
        output: list[TranscriptChunk] = []
        for chunk in chunks:
            start = chunk.get("start", getattr(chunk, "start", 0))
            end = chunk.get("end", getattr(chunk, "end", 0))
            text = (chunk.get("text", "") or getattr(chunk, "text", "") or "").strip()
            if text:
                output.append(TranscriptChunk(start_s=float(start), end_s=float(end), text=text))
        return output
    finally:
        try:
            os.unlink(audio_path)
        except OSError:
            pass


def map_questions_to_time_ranges(
    questions: tuple[str, ...],
    transcript_chunks: list[TranscriptChunk],
    gemini_api_key: str,
    duration_s: float,
    cancel_event: Optional[threading.Event] = None,
) -> list[QuestionWindow]:
    def fallback() -> list[QuestionWindow]:
        step = duration_s / 4 if duration_s else 0
        return [
            QuestionWindow(
                question=(questions[i] if i < len(questions) else f"Q{i + 1}") or f"Q{i + 1}",
                start_s=i * step,
                end_s=(i + 1) * step,
            )
            for i in range(4)
        ]

    if _cancelled(cancel_event):
        return []
    if not transcript_chunks or len(questions) < 4:
        return fallback()

    transcript_blob = "\n".join(
        f"[{chunk.start_s:.1f}s - {chunk.end_s:.1f}s] {chunk.text}" for chunk in transcript_chunks
    )
    q_lines = "\n".join(f"Q{i + 1}: {question}" for i, question in enumerate(questions[:4]))
    prompt = (
        "Assign a continuous answer window to each of the 4 interview questions.\n"
        "Return valid JSON only.\n"
        '[{"question": "Q1 text", "start_s": 0.0, "end_s": 12.0}, ...]\n'
        f"Video duration: {duration_s:.1f} seconds.\n\n"
        f"Transcript:\n{transcript_blob}\n\n"
        f"Questions:\n{q_lines}\n"
    )

    try:
        client = genai.Client(api_key=gemini_api_key)
        text = _retryable_gemini_call(client, prompt, cancel_event)
        if "```" in text:
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        data = json.loads(text)
    except Exception:
        return fallback()

    if not isinstance(data, list) or len(data) < 4:
        return fallback()

    windows: list[QuestionWindow] = []
    for idx, item in enumerate(data[:4]):
        if not isinstance(item, dict):
            windows.append(fallback()[idx])
            continue
        start_s = float(item.get("start_s", 0))
        end_s = float(item.get("end_s", duration_s))
        question = item.get("question") or questions[idx]
        start_s = max(0, min(start_s, max(duration_s - 0.5, 0)))
        end_s = max(start_s + 1, min(end_s, duration_s)) if duration_s else start_s + 1
        windows.append(QuestionWindow(question=question, start_s=start_s, end_s=end_s))
    return windows


def _gemini_pick_one_for_question(
    client: genai.Client,
    question_text: str,
    batch_frames: list[SelectedFrame],
    cancel_event: Optional[threading.Event],
) -> int:
    prompt = (
        "Select the single strongest frame for this interview question.\n"
        f"Question: {question_text}\n"
        "Prefer an expressive, flattering reaction frame with sharp eyes and a clean crop.\n"
        f"There are {len(batch_frames)} frames indexed 0 to {len(batch_frames) - 1}.\n"
        'Return only JSON: {"index": 0}'
    )
    contents = [prompt]
    for idx, frame in enumerate(batch_frames):
        contents.append(f"\nFrame {idx}:")
        contents.append(_frame_part(frame.frame_bgr))
    text = _retryable_gemini_call(client, contents, cancel_event)
    payload = _extract_json_block(text, "index")
    if payload and isinstance(payload.get("index"), int):
        idx = payload["index"]
        if 0 <= idx < len(batch_frames):
            return idx
    for idx in map(int, re.findall(r"\b\d+\b", text)):
        if 0 <= idx < len(batch_frames):
            return idx
    return 0


def _placeholder_frame(video_path: str, start_s: float) -> Optional[SelectedFrame]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 1
    frame_index = int(start_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return SelectedFrame(frame_bgr=frame.copy(), score=0.0, frame_index=frame_index)


def run_pipeline(
    video_path: str,
    options: ExtractionOptions,
    progress_cb: StatusCallback = None,
    cancel_event: Optional[threading.Event] = None,
) -> list[SelectedFrame]:
    if options.use_question_mode:
        meta = get_video_metadata(video_path)
        _emit(progress_cb, "Transcribing audio with Whisper...", 6)
        try:
            chunks = transcribe_video_to_chunks(video_path, options.openai_key, cancel_event)
        except Exception:
            chunks = []
        if _cancelled(cancel_event):
            return []
        _emit(progress_cb, "Mapping questions to answers...", 18)
        windows = map_questions_to_time_ranges(options.questions[:4], chunks, options.gemini_key, meta.duration_s, cancel_event)
        if _cancelled(cancel_event):
            return []
        client = genai.Client(api_key=options.gemini_key)
        results: list[SelectedFrame] = []
        for idx, window in enumerate(windows[:4], start=1):
            _emit(progress_cb, f"Q{idx}: scanning candidate frames...", 22 + (idx - 1) * 15)
            raw = extract_sharp_frames_in_range(video_path, window.start_s, window.end_s, options.sample_every, cancel_event)
            candidates = temporal_diversity_filter(raw, options.min_gap, options.top_n) or raw[: options.top_n]
            if not candidates:
                placeholder = _placeholder_frame(video_path, window.start_s)
                if placeholder:
                    results.append(
                        SelectedFrame(
                            frame_bgr=placeholder.frame_bgr,
                            score=placeholder.score,
                            frame_index=placeholder.frame_index,
                            question_label=window.question,
                        )
                    )
                continue
            _emit(progress_cb, f"Q{idx}: selecting best moment...", 29 + (idx - 1) * 15)
            picked = candidates[_gemini_pick_one_for_question(client, window.question, candidates, cancel_event)]
            results.append(
                SelectedFrame(
                    frame_bgr=picked.frame_bgr,
                    score=picked.score,
                    frame_index=picked.frame_index,
                    question_label=window.question,
                )
            )
        _emit(progress_cb, f"Selected {len(results)} frames.", 100)
        return results

    _emit(progress_cb, "Scanning the full video for sharp frames...", 0)

    def on_frame_progress(current: int, total: int) -> None:
        _emit(progress_cb, f"Scanning frames {current:,}/{total:,}...", min(current / max(total, 1) * 48, 48))

    raw = extract_sharp_frames(video_path, options.sample_every, cancel_event, on_frame_progress)
    if _cancelled(cancel_event):
        return []
    candidates = temporal_diversity_filter(raw, options.min_gap, options.top_n)
    _emit(progress_cb, f"Curating {len(candidates)} candidates with Gemini...", 50)
    selected = gemini_select(candidates, options.gemini_key, options.n_select, progress_cb, cancel_event)
    _emit(progress_cb, f"Selected {len(selected)} frames.", 100)
    return selected

