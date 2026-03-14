#!/usr/bin/env python3
"""
Candid Photo Strip — question-aware pipeline
  Stage 1 : Extract audio from video (ffmpeg)
  Stage 2 : Transcribe audio → timestamped chunks (Gemini Files API)
  Stage 3 : Map 4 user questions to time windows (gap detection + equal-split fallback)
  Stage 4 : For each window — OpenCV sharpness filter → Gemini picks 1 best frame
  UI      : Tkinter vertical photo strip (4 frames + question text per cell)
"""

import cv2
import io
import json
import os
import platform
import re
import subprocess
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import google.genai as genai
from PIL import Image, ImageTk

# ── Config ──────────────────────────────────────────────────────────────────
NUM_QUESTIONS     = 4
TOP_SHARP_PER_SEG = 20       # candidate frames per segment sent to Gemini
SAMPLE_EVERY_N    = 15       # sample every N frames (denser for short windows)
THUMB_W, THUMB_H  = 320, 240 # thumbnail size for Gemini calls
STRIP_THUMB_W     = 520      # thumbnail width in results strip
STRIP_THUMB_H     = 390      # thumbnail height in results strip
GEMINI_MODEL      = "gemini-2.5-flash"
MAX_RETRIES       = 3
RETRY_DELAY       = 2.0
# ─────────────────────────────────────────────────────────────────────────────

# ── Palette — pure black & white ─────────────────────────────────────────────
BG      = "#000000"
BG2     = "#0a0a0a"
BORDER  = "#1c1c1c"
MUTED   = "#444444"
TEXT    = "#aaaaaa"
WHITE   = "#ffffff"
# ─────────────────────────────────────────────────────────────────────────────

HOW_IT_WORKS_1 = (
    "Enter your 4 interview questions. The app transcribes the video audio "
    "with Gemini and detects where each question's answer begins and ends. "
    "If transcription is unclear, the video is split into equal windows."
)
HOW_IT_WORKS_2 = (
    "For each of the 4 answer segments, OpenCV scores frames by sharpness "
    "and the top candidates are sent to Gemini with the specific question as "
    "context — so each panel gets the most expressive, relevant frame."
)


# ── Stage 1 helpers ──────────────────────────────────────────────────────────
def extract_sharp_frames(video_path: str, sample_every: int, fps: float,
                         start_s: float = 0.0, end_s: float = float("inf"),
                         progress_cb=None,
                         cancel_event: threading.Event = None):
    """Extract and rank frames by Laplacian sharpness within [start_s, end_s]."""
    cap = cv2.VideoCapture(video_path)
    start_frame = int(start_s * fps)
    end_frame   = int(end_s   * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames, scores, orig_indices = [], [], []
    idx = start_frame
    span = max(end_frame - start_frame, 1)

    while True:
        if cancel_event and cancel_event.is_set():
            cap.release()
            return []
        if idx >= end_frame:
            break
        ret, frame = cap.read()
        if not ret:
            break
        if (idx - start_frame) % sample_every == 0:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            frames.append(frame.copy())
            scores.append(score)
            orig_indices.append(idx)
            if progress_cb:
                progress_cb(min((idx - start_frame) / span * 100, 100))
        idx += 1

    cap.release()
    order = np.argsort(scores)[::-1]
    return [(frames[i], scores[i], orig_indices[i]) for i in order]


def get_video_metadata(video_path: str) -> dict:
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 1
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "frames": frames,
            "duration_s": frames / fps, "width": width, "height": height}


# ── Audio extraction ─────────────────────────────────────────────────────────
def extract_audio_to_wav(video_path: str,
                         cancel_event: threading.Event = None) -> str:
    """Extract mono 16 kHz PCM WAV from video using ffmpeg. Returns temp path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    out_path = tmp.name

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        out_path,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while proc.poll() is None:
        if cancel_event and cancel_event.is_set():
            proc.kill()
            raise RuntimeError("Cancelled during audio extraction.")
        time.sleep(0.2)

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}.")
    return out_path


# ── Transcription ─────────────────────────────────────────────────────────────
def transcribe_audio(client: genai.Client, audio_path: str,
                     cancel_event: threading.Event = None) -> list[dict]:
    """
    Upload WAV to Gemini Files API and get timestamped transcript chunks.
    Returns list of {"start_s": float, "end_s": float, "text": str}.
    Returns [] on any failure — caller uses equal-split fallback.
    """
    try:
        uploaded = client.files.upload(
            file=audio_path,
            config=genai.types.UploadFileConfig(mime_type="audio/wav"),
        )

        # Poll until ACTIVE
        for _ in range(30):
            if cancel_event and cancel_event.is_set():
                return []
            f = client.files.get(name=uploaded.name)
            if f.state.name == "ACTIVE":
                break
            time.sleep(2)
        else:
            return []

        prompt = (
            "You are a transcription engine. Listen to the attached audio and "
            "produce a timestamped transcript broken into short chunks of 2–4 seconds each.\n"
            "Return ONLY a JSON array — no prose, no markdown fences. Each element:\n"
            '{"start_s": <float>, "end_s": <float>, "text": "<spoken words>"}\n'
            "Include silent ranges as elements with empty text string.\n"
            "Sort by start_s ascending."
        )

        audio_part = genai.types.Part.from_uri(
            file_uri=uploaded.uri,
            mime_type="audio/wav",
        )

        for attempt in range(MAX_RETRIES):
            if cancel_event and cancel_event.is_set():
                return []
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[prompt, audio_part],
                )
                text = response.text.strip()
                text = re.sub(r"^```[a-z]*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
                chunks = json.loads(text)
                if isinstance(chunks, list) and all("start_s" in c for c in chunks):
                    return chunks
            except Exception:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
    except Exception:
        pass
    return []


# ── Segmentation ─────────────────────────────────────────────────────────────
def equal_split_windows(questions: list[str], duration_s: float) -> list[dict]:
    """Divide duration evenly into NUM_QUESTIONS windows."""
    seg = duration_s / len(questions)
    return [
        {"q_idx": i, "question": q,
         "start_s": i * seg, "end_s": (i + 1) * seg}
        for i, q in enumerate(questions)
    ]


def map_transcript_to_windows(chunks: list[dict], questions: list[str],
                               video_duration_s: float) -> list[dict]:
    """
    Find the N-1 largest silence gaps to split transcript into N question windows.
    Falls back to equal_split_windows if gaps are too small or unclear.
    """
    n = len(questions)
    if not chunks or n < 2:
        return equal_split_windows(questions, video_duration_s)

    # Compute gaps between consecutive chunks
    gaps = []
    for i in range(len(chunks) - 1):
        gap_start = chunks[i]["end_s"]
        gap_end   = chunks[i + 1]["start_s"]
        gap_dur   = gap_end - gap_start
        if gap_dur > 0:
            gaps.append((gap_dur, gap_start))  # (duration, position)

    # Need at least n-1 gaps of meaningful size
    gaps.sort(reverse=True)
    if len(gaps) < n - 1 or gaps[n - 2][0] < 0.3:
        return equal_split_windows(questions, video_duration_s)

    # Pick the n-1 largest gaps as boundaries, sorted by position
    boundary_positions = sorted(g[1] for g in gaps[:n - 1])

    # Build windows
    boundaries = [0.0] + boundary_positions + [video_duration_s]
    windows = []
    for i, q in enumerate(questions):
        windows.append({
            "q_idx":   i,
            "question": q,
            "start_s": boundaries[i],
            "end_s":   boundaries[i + 1],
        })
    return windows


# ── Per-segment Gemini selection ──────────────────────────────────────────────
def gemini_pick_one_frame(client: genai.Client, candidates: list,
                          question_text: str,
                          cancel_event: threading.Event = None) -> int:
    """
    Send up to TOP_SHARP_PER_SEG candidate frames to Gemini with a
    question-specific prompt. Returns index into candidates of the best frame.
    Returns 0 (sharpest) as safe fallback.
    """
    if not candidates:
        return 0

    prompt = (
        f"You are selecting the single best photo for one panel of a printed photo strip.\n"
        f"The interview question being answered in these frames is:\n"
        f'  "{question_text}"\n\n'
        f"There are {len(candidates)} frames indexed 0 to {len(candidates) - 1}.\n\n"
        f"Choose the ONE frame that best captures the subject at a meaningful, "
        f"expressive, or emotionally resonant moment while answering that question.\n\n"
        f"Hard requirements (disqualify any frame that fails):\n"
        f"- Eyes must be open and in sharp focus.\n"
        f"- No heavy motion blur on the face.\n"
        f"- Face must not be cut off at the edge or turned more than ~90° away.\n\n"
        f"Preference order:\n"
        f"1. Genuine expression — smile reaching the eyes, strong reaction, thoughtful look.\n"
        f"2. Clear, confident eye contact with the camera.\n"
        f"3. Neutral but sharp, flattering, and well-composed frame.\n\n"
        f'Return ONLY a JSON object: {{"index": <integer>}}'
    )

    parts = [prompt]
    for i, (frame, _score, _orig) in enumerate(candidates):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
        parts.append(f"\nFrame {i}:")
        parts.append(genai.types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))

    for attempt in range(MAX_RETRIES):
        if cancel_event and cancel_event.is_set():
            return 0
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL, contents=parts
            )
            text = response.text.strip()
            m = re.search(r'\{"index"\s*:\s*(\d+)\}', text)
            if m:
                idx = int(m.group(1))
                if 0 <= idx < len(candidates):
                    return idx
            # Fallback: first integer in response
            nums = re.findall(r'\b\d+\b', text)
            if nums:
                idx = int(nums[0])
                if 0 <= idx < len(candidates):
                    return idx
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return 0


# ── Zoom window ───────────────────────────────────────────────────────────────
class ZoomWindow(tk.Toplevel):
    def __init__(self, parent, frame_bgr, title=""):
        super().__init__(parent)
        self.title(title)
        self.configure(bg=BG)
        self.resizable(True, True)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        img.thumbnail((int(sw * 0.85), int(sh * 0.85)), Image.LANCZOS)

        self._tk_img = ImageTk.PhotoImage(img)
        lbl = tk.Label(self, image=self._tk_img, bg=BG)
        lbl.pack(padx=0, pady=0)
        self.bind("<Escape>", lambda _: self.destroy())
        lbl.bind("<Button-1>", lambda _: self.destroy())


# ── Main application ──────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Candid Photo Strip")
        self.configure(bg=BG)
        self.minsize(700, 720)
        self._tk_images:      list[ImageTk.PhotoImage] = []
        self._frame_data:     list[dict]               = []
        self._last_video_path: str | None              = None
        self._cancel_event   = threading.Event()
        self._last_save_dir  = os.path.expanduser("~/Pictures")
        self._question_labels: list[tk.Label]          = []
        self._build_ui()

    # ── build ─────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_header()
        self._build_how_it_works()
        self._build_controls()
        self._build_progress()
        self._results_section = tk.Frame(self, bg=BG)
        tk.Frame(self._results_section, bg=BORDER, height=1).pack(fill="x")
        self._build_results_header(self._results_section)
        self._build_grid(self._results_section)

    def _build_header(self):
        hdr = tk.Frame(self, bg=BG, pady=22)
        hdr.pack(fill="x", padx=32)

        tk.Label(
            hdr, text="Candid Photo Strip",
            font=("Helvetica", 22, "bold"),
            fg=WHITE, bg=BG, anchor="w",
        ).pack(side="left")

        tk.Label(
            hdr, text=GEMINI_MODEL,
            font=("Helvetica", 10),
            fg=MUTED, bg=BG,
        ).pack(side="right", pady=6)

        self._separator()

    def _build_how_it_works(self):
        outer = tk.Frame(self, bg=BG, padx=32, pady=16)
        outer.pack(fill="x")

        tk.Label(
            outer, text="HOW IT WORKS",
            font=("Helvetica", 8, "bold"),
            fg=MUTED, bg=BG, anchor="w",
        ).pack(anchor="w", pady=(0, 6))

        cols = tk.Frame(outer, bg=BG)
        cols.pack(fill="x")
        cols.columnconfigure(0, weight=1)
        cols.columnconfigure(1, weight=1)

        self._info_card(cols, col=0, number="01",
                        title="Audio transcription & segmentation",
                        body=HOW_IT_WORKS_1)
        self._info_card(cols, col=1, number="02",
                        title="Per-question Gemini frame selection",
                        body=HOW_IT_WORKS_2)

        self._separator(pady=(16, 0))

    def _info_card(self, parent, col, number, title, body):
        card = tk.Frame(parent, bg=BG, padx=(0 if col == 0 else 24), pady=0)
        card.grid(row=0, column=col, sticky="nsew")

        tk.Label(card, text=number, font=("Helvetica", 28, "bold"),
                 fg=BORDER, bg=BG, anchor="w").pack(anchor="w")
        tk.Label(card, text=title, font=("Helvetica", 11, "bold"),
                 fg=WHITE, bg=BG, anchor="w").pack(anchor="w", pady=(0, 5))
        tk.Label(card, text=body, font=("Helvetica", 9),
                 fg=TEXT, bg=BG, anchor="w",
                 justify="left", wraplength=360).pack(anchor="w")

    def _build_controls(self):
        ctrl = tk.Frame(self, bg=BG, padx=32, pady=16)
        ctrl.pack(fill="x")

        self.video_var = tk.StringVar()
        self.key_var   = tk.StringVar(value=os.environ.get("GEMINI_API_KEY", ""))

        self._field(ctrl, "VIDEO FILE", self.video_var, browse=self._browse_video)
        self._field(ctrl, "GEMINI API KEY", self.key_var, secret=True)

        self.meta_var = tk.StringVar()
        tk.Label(ctrl, textvariable=self.meta_var,
                 fg=MUTED, bg=BG, font=("Helvetica", 8),
                 anchor="w").pack(anchor="w", pady=(0, 8))

        # 4 question fields
        tk.Label(ctrl, text="INTERVIEW QUESTIONS",
                 font=("Helvetica", 7, "bold"),
                 fg=MUTED, bg=BG, anchor="w").pack(anchor="w", pady=(4, 4))

        self.question_vars = []
        for i in range(NUM_QUESTIONS):
            var = tk.StringVar()
            self.question_vars.append(var)
            self._field(ctrl, f"Q{i + 1}", var)

        # buttons
        btn_row = tk.Frame(ctrl, bg=BG)
        btn_row.pack(anchor="w", pady=(4, 0))

        self.run_btn = tk.Button(
            btn_row, text="Extract",
            command=self._run,
            bg=WHITE, fg=BG,
            font=("Helvetica", 10, "bold"),
            relief="flat", padx=20, pady=8, cursor="hand2",
            activebackground="#dddddd", activeforeground=BG,
        )
        self.run_btn.pack(side="left", padx=(0, 8))

        self.cancel_btn = tk.Button(
            btn_row, text="Cancel",
            command=self._cancel,
            bg=BG, fg=MUTED,
            font=("Helvetica", 10),
            relief="flat", padx=16, pady=8, cursor="hand2",
            state="disabled",
            activebackground=BG2, activeforeground=WHITE,
        )
        self.cancel_btn.pack(side="left")

    def _field(self, parent, label, var, browse=None, secret=False):
        wrap = tk.Frame(parent, bg=BG)
        wrap.pack(fill="x", pady=(0, 10))

        tk.Label(wrap, text=label, font=("Helvetica", 7, "bold"),
                 fg=MUTED, bg=BG, anchor="w").pack(anchor="w", pady=(0, 3))

        row = tk.Frame(wrap, bg=BG)
        row.pack(fill="x")

        kw = dict(show="•") if secret else {}
        entry = tk.Entry(
            row, textvariable=var,
            bg=BG2, fg=WHITE, insertbackground=WHITE,
            relief="flat", bd=0,
            font=("Helvetica", 10),
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=WHITE,
            **kw,
        )
        entry.pack(side="left", fill="x", expand=True, ipady=7, padx=(0, 1))

        if browse:
            tk.Button(
                row, text="Browse",
                command=browse,
                bg=BG2, fg=TEXT,
                font=("Helvetica", 9), relief="flat",
                padx=12, pady=7, cursor="hand2",
                highlightthickness=1,
                highlightbackground=BORDER,
                activebackground=BORDER, activeforeground=WHITE,
            ).pack(side="left")

    def _build_progress(self):
        prog = tk.Frame(self, bg=BG, padx=32)
        prog.pack(fill="x", pady=(0, 4))

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(prog, textvariable=self.status_var,
                 fg=MUTED, bg=BG, font=("Helvetica", 8),
                 anchor="w").pack(anchor="w", pady=(0, 4))

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TProgressbar",
                        troughcolor=BG2, background=WHITE,
                        bordercolor=BG, lightcolor=WHITE,
                        darkcolor=WHITE, thickness=2)
        self.progress = ttk.Progressbar(prog, length=800,
                                        mode="determinate",
                                        style="TProgressbar")
        self.progress.pack(fill="x", pady=(0, 8))

    def _build_results_header(self, parent):
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", padx=32, pady=(10, 4))

        tk.Label(row, text="RESULTS",
                 font=("Helvetica", 7, "bold"),
                 fg=MUTED, bg=BG).pack(side="left")

        tk.Label(row, text="click any frame to zoom  ·  esc to close",
                 font=("Helvetica", 7),
                 fg=MUTED, bg=BG).pack(side="left", padx=12)

        self.dl_all_btn = tk.Button(
            row, text="Save All",
            command=self._download_all,
            bg=BG, fg=MUTED,
            font=("Helvetica", 8), relief="flat",
            padx=10, pady=2, cursor="hand2",
            state="disabled",
            activebackground=BG2, activeforeground=WHITE,
        )
        self.dl_all_btn.pack(side="right")

    def _build_grid(self, parent):
        wrapper = tk.Frame(parent, bg=BG)
        wrapper.pack(fill="both", expand=True, padx=24, pady=(0, 24))

        canvas = tk.Canvas(wrapper, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(wrapper, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self._grid_frame = tk.Frame(canvas, bg=BG)
        self._canvas_window = canvas.create_window(
            (0, 0), window=self._grid_frame, anchor="nw"
        )

        self._grid_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfig(self._canvas_window, width=e.width),
        )

        if platform.system() == "Darwin":
            canvas.bind_all("<MouseWheel>",
                            lambda e: canvas.yview_scroll(-1 * e.delta, "units"))
        else:
            canvas.bind_all("<MouseWheel>",
                            lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
            canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
            canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        self._cells:       list[tk.Label]  = []
        self._captions:    list[tk.Label]  = []
        self._dl_btns:     list[tk.Button] = []
        self._cell_frames: list[tk.Frame]  = []

    def _separator(self, pady=(0, 0)):
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", pady=pady)

    # ── actions ───────────────────────────────────────────────────────────────
    def _browse_video(self):
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
                       ("All files", "*.*")],
        )
        if path:
            self.video_var.set(path)
            self._load_metadata(path)

    def _load_metadata(self, path):
        try:
            m = get_video_metadata(path)
            dur = m["duration_s"]
            mins, secs = divmod(int(dur), 60)
            self.meta_var.set(
                f"{m['width']}×{m['height']}  ·  {m['fps']:.2f} fps  "
                f"·  {mins}m {secs:02d}s  ·  {m['frames']:,} frames"
            )
        except Exception:
            self.meta_var.set("")

    def _cancel(self):
        self._cancel_event.set()
        self.cancel_btn.config(state="disabled")
        self._update("Cancelling…", self.progress["value"])

    def _run(self):
        video = self.video_var.get().strip()
        key   = self.key_var.get().strip()
        if not video:
            messagebox.showerror("Missing input", "Please select a video file.")
            return
        if not os.path.isfile(video):
            messagebox.showerror("Not found", f"File not found:\n{video}")
            return
        if not key:
            messagebox.showerror("Missing key", "Please enter your Gemini API key.")
            return

        questions = [v.get().strip() for v in self.question_vars]
        if any(q == "" for q in questions):
            messagebox.showerror("Missing questions",
                                 "Please fill in all 4 interview questions.")
            return

        self._cancel_event.clear()
        self.run_btn.config(state="disabled")
        self.cancel_btn.config(state="normal", fg=TEXT)
        self.dl_all_btn.config(state="disabled")
        self._clear_grid()
        threading.Thread(target=self._pipeline,
                         args=(video, key, questions), daemon=True).start()

    def _pipeline(self, video: str, key: str, questions: list[str]):
        audio_path = None
        try:
            client = genai.Client(api_key=key)
            meta   = get_video_metadata(video)
            fps    = meta["fps"]
            dur    = meta["duration_s"]

            # Stage 1: extract audio
            self._update("01 / extracting audio…", 5)
            try:
                audio_path = extract_audio_to_wav(video, self._cancel_event)
            except Exception as e:
                self._update(f"Audio extraction failed ({e}), using equal splits.", 10)
                audio_path = None

            if self._cancel_event.is_set():
                self._update("Cancelled.", 0)
                return

            # Stage 2: transcribe + segment
            windows = None
            if audio_path:
                self._update("02 / transcribing audio…", 15)
                try:
                    chunks = transcribe_audio(client, audio_path, self._cancel_event)
                    if chunks:
                        windows = map_transcript_to_windows(chunks, questions, dur)
                except Exception as e:
                    self._update(f"Transcription failed ({e}), using equal splits.", 20)

            if windows is None:
                windows = equal_split_windows(questions, dur)

            if self._cancel_event.is_set():
                self._update("Cancelled.", 0)
                return

            # Stages 3+4: per-segment extraction + Gemini selection
            results = []
            for seg_idx, win in enumerate(windows):
                if self._cancel_event.is_set():
                    self._update("Cancelled.", 0)
                    return

                base_pct = 30 + seg_idx * 17
                self._update(
                    f"03 / Q{seg_idx + 1} — scanning "
                    f"{win['start_s']:.1f}s – {win['end_s']:.1f}s…",
                    base_pct,
                )
                candidates = extract_sharp_frames(
                    video, SAMPLE_EVERY_N, fps,
                    start_s=win["start_s"], end_s=win["end_s"],
                    cancel_event=self._cancel_event,
                )[:TOP_SHARP_PER_SEG]

                if not candidates:
                    self._update(f"Warning: no frames found in Q{seg_idx + 1} window.", base_pct + 4)
                    continue

                self._update(
                    f"04 / Q{seg_idx + 1} — asking Gemini "
                    f"({len(candidates)} candidates)…",
                    base_pct + 8,
                )
                pick = gemini_pick_one_frame(
                    client, candidates, win["question"], self._cancel_event
                )
                frame, score, orig_idx = candidates[pick]
                results.append({
                    "frame":    frame,
                    "score":    score,
                    "orig_idx": orig_idx,
                    "question": win["question"],
                    "label":    f"Q{seg_idx + 1}",
                })

            if self._cancel_event.is_set():
                self._update("Cancelled.", 0)
                return

            self._update(f"{len(results)} frames selected.", 100)
            self.after(0, lambda r=results: self._show_strip(r, video))

        except Exception as exc:
            self.after(0, lambda e=exc: messagebox.showerror("Error", str(e)))
            self._update(f"Error: {exc}", 0)
        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except OSError:
                    pass
            self.after(0, lambda: self.run_btn.config(state="normal"))
            self.after(0, lambda: self.cancel_btn.config(state="disabled", fg=MUTED))

    def _zoom(self, idx: int):
        if idx >= len(self._frame_data):
            return
        item = self._frame_data[idx]
        ZoomWindow(self, item["frame"],
                   title=f"{item['label']}  ·  frame {item['orig_idx']}  ·  sharpness {item['score']:,.0f}")

    def _save_dialog(self, initial_file: str):
        path = filedialog.asksaveasfilename(
            title="Save frame",
            initialdir=self._last_save_dir,
            initialfile=initial_file,
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All files", "*.*")],
        )
        if path:
            self._last_save_dir = os.path.dirname(path)
        return path

    def _write_frame(self, path: str, frame_bgr):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".jpg", ".jpeg"):
            cv2.imwrite(path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(path, frame_bgr)

    def _download_single(self, idx: int):
        if idx >= len(self._frame_data):
            return
        item = self._frame_data[idx]
        path = self._save_dialog(f"{item['label']}_frame_{item['orig_idx']}.jpg")
        if path:
            self._write_frame(path, item["frame"])
            self._update(f"Saved → {os.path.basename(path)}", self.progress["value"])

    def _download_all(self):
        if not self._frame_data:
            return
        video_path = self._last_video_path or self.video_var.get().strip()
        if not video_path or not os.path.isfile(video_path):
            messagebox.showerror(
                "Save All",
                "Video path is missing or invalid. Re-run extraction and try again.",
            )
            return
        video_dir      = os.path.dirname(os.path.abspath(video_path))
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        folder = os.path.join(video_dir, video_basename)
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError as e:
            messagebox.showerror("Save All", f"Could not create folder:\n{e}")
            return
        for item in self._frame_data:
            self._write_frame(
                os.path.join(folder, f"{item['label']}_frame_{item['orig_idx']}.jpg"),
                item["frame"],
            )
        self._last_save_dir = folder
        self._update(
            f"Saved {len(self._frame_data)} frames → {folder}",
            self.progress["value"],
        )
        messagebox.showinfo(
            "Save All",
            f"Saved {len(self._frame_data)} frames to:\n{folder}",
        )

    # ── helpers ───────────────────────────────────────────────────────────────
    def _update(self, msg, value):
        if msg is not None:
            self.after(0, lambda m=msg: self.status_var.set(m))
        self.after(0, lambda v=value: self.progress.config(value=v))

    def _clear_grid(self):
        self._tk_images.clear()
        self._frame_data.clear()
        self._question_labels.clear()
        for outer in self._cell_frames:
            outer.destroy()
        self._cell_frames.clear()
        self._cells.clear()
        self._captions.clear()
        self._dl_btns.clear()
        self._results_section.pack_forget()

    def _build_grid_cells(self, n: int):
        """Build exactly n cells in a 1-column vertical strip."""
        for outer in self._cell_frames:
            outer.destroy()
        self._cell_frames.clear()
        self._cells.clear()
        self._captions.clear()
        self._dl_btns.clear()
        self._question_labels.clear()

        self._grid_frame.columnconfigure(0, weight=1)

        for i in range(n):
            outer = tk.Frame(self._grid_frame, bg=BORDER)
            outer.grid(row=i, column=0, padx=6, pady=6, sticky="nsew")
            self._cell_frames.append(outer)

            inner = tk.Frame(outer, bg=BG2)
            inner.pack(fill="both", expand=True, padx=1, pady=1)

            img_lbl = tk.Label(inner, bg=BG2, text="", relief="flat", cursor="hand2")
            img_lbl.pack(fill="both", expand=True)
            img_lbl.bind("<Button-1>", lambda e, idx=i: self._zoom(idx))

            footer = tk.Frame(inner, bg=BG2, pady=5)
            footer.pack(fill="x", padx=8)

            cap_lbl = tk.Label(footer, bg=BG2, fg=MUTED, text="—",
                               font=("Helvetica", 7), anchor="w")
            cap_lbl.pack(side="left")

            dl_btn = tk.Button(
                footer, text="Save",
                command=lambda idx=i: self._download_single(idx),
                bg=BG2, fg=MUTED,
                font=("Helvetica", 7), relief="flat",
                padx=6, pady=0, cursor="hand2",
                state="normal",
                activebackground=BG2, activeforeground=WHITE,
            )
            dl_btn.pack(side="right")

            q_lbl = tk.Label(inner, bg=BG2, fg=TEXT,
                             text="", font=("Helvetica", 10, "italic"),
                             anchor="w", justify="left", wraplength=500)
            q_lbl.pack(fill="x", padx=8, pady=(0, 8))

            self._cells.append(img_lbl)
            self._captions.append(cap_lbl)
            self._dl_btns.append(dl_btn)
            self._question_labels.append(q_lbl)

    def _show_strip(self, results: list[dict], video_path: str | None = None):
        if video_path:
            self._last_video_path = video_path
        self._results_section.pack(fill="both", expand=True)
        n = len(results)
        self._build_grid_cells(n)
        self._tk_images.clear()
        self._frame_data = list(results)

        for i, item in enumerate(results):
            lbl     = self._cells[i]
            cap_lbl = self._captions[i]
            q_lbl   = self._question_labels[i]
            dl_btn  = self._dl_btns[i]

            rgb = cv2.cvtColor(item["frame"], cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img.thumbnail((STRIP_THUMB_W, STRIP_THUMB_H), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(img)
            self._tk_images.append(tk_img)

            lbl.config(image=tk_img, text="")
            cap_lbl.config(
                text=f"{item['label']}  ·  frame {item['orig_idx']}  ·  sharpness {item['score']:,.0f}",
                fg=TEXT,
            )
            q_lbl.config(text=item["question"])
            dl_btn.config(state="normal", fg=TEXT)

        self.dl_all_btn.config(state="normal", fg=TEXT)


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()
