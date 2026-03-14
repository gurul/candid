#!/usr/bin/env python3
"""
Video Frame Extractor — two-stage pipeline
  Stage 1 : OpenCV Laplacian-variance sharpness filter + temporal diversity
  Stage 2 : Gemini visual selection (chunked batches with retry)
  UI      : Tkinter 2×5 scrollable grid showing the 10 best frames
            with per-frame Download buttons, click-to-zoom, cancel support
"""

import cv2
import io
import json
import os
import platform
import re
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import google.generativeai as genai
from PIL import Image, ImageTk

# ── Config ──────────────────────────────────────────────────────────────────
SAMPLE_EVERY_N   = 30
TOP_N_SHARP      = 150
GEMINI_SELECTS   = 10
DISPLAY_N        = 50   # max pre-built grid cells (covers full selects_var range)
THUMB_W, THUMB_H = 320, 240
GEMINI_MODEL     = "gemini-2.5-flash"
GEMINI_BATCH     = 30
TEMPORAL_GAP     = 90
MAX_RETRIES      = 3
RETRY_DELAY      = 2.0
# ─────────────────────────────────────────────────────────────────────────────

# ── Palette — pure black & white ─────────────────────────────────────────────
BG      = "#000000"   # true black
BG2     = "#0a0a0a"   # near-black panels
BORDER  = "#1c1c1c"   # subtle borders
MUTED   = "#444444"   # secondary text
TEXT    = "#aaaaaa"   # body text
WHITE   = "#ffffff"   # primary / interactive
# ─────────────────────────────────────────────────────────────────────────────

HOW_IT_WORKS = (
    "Stage 1 — OpenCV samples 1 frame every N frames and scores each one with "
    "Laplacian-variance sharpness. A temporal-diversity filter then prunes "
    "near-duplicate frames so candidates are spread across the whole video. "
    "The top candidates move to Stage 2 — Google Gemini evaluates them visually "
    "(in batches to stay within token limits) and picks the 10 most expressive, "
    "flattering, and narratively cohesive frames using professional photo-editor "
    "criteria: open eyes, genuine Duchenne smiles, no motion blur, and story arc."
)


# ── Stage 1 : sharpness scoring ──────────────────────────────────────────────
def extract_sharp_frames(video_path: str, sample_every: int, progress_cb=None,
                         cancel_event: threading.Event = None):
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, scores, orig_indices = [], [], []
    idx = 0
    while True:
        if cancel_event and cancel_event.is_set():
            cap.release()
            return []
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_every == 0:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            frames.append(frame.copy())
            scores.append(score)
            orig_indices.append(idx)
            if progress_cb:
                progress_cb(min(idx / max(total, 1) * 48, 48))
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


def temporal_diversity_filter(frames, min_gap: int, limit: int):
    if min_gap <= 0:
        return frames[:limit]
    kept, last_orig = [], -min_gap - 1
    for item in frames:
        _f, _s, orig_idx = item
        if orig_idx - last_orig >= min_gap:
            kept.append(item)
            last_orig = orig_idx
            if len(kept) >= limit:
                break
    return kept


# ── Stage 2 : Gemini selection ───────────────────────────────────────────────
def _gemini_pick_batch(model, batch_frames, batch_offset: int,
                       total_select: int, cancel_event) -> list[int]:
    prompt = (
        "You are a professional photo editor curating frames for a vertical photo strip. "
        "Your job is to select the most expressive, flattering, and narratively cohesive frames "
        "using the same criteria as a high-end event photographer.\n"
        "\n"
        "TECHNICAL REQUIREMENTS (hard filters — disqualify any frame that fails these):\n"
        "- Eyes of the primary subject must be sharp and in focus (eyes are the absolute priority).\n"
        "- No heavy accidental motion blur on faces.\n"
        "- No faces cut off at the frame edge or turned more than ~90 degrees away.\n"
        "- No faces more than 30% obscured by hair, hands, or other objects.\n"
        "- Skin tones should look natural — avoid blown highlights or harsh shadows on the face.\n"
        "\n"
        "FLATTERY & DIGNITY FILTERS (prefer frames that pass all of these):\n"
        "- Eyes are clearly open — avoid blinks and 'blink-adjacent' half-closed eyes.\n"
        "- Avoid mid-chew, mid-sentence, or mid-drink frames where the mouth looks distorted.\n"
        "- Prefer frames where the jawline is defined (chin slightly forward/down, not pulled back).\n"
        "- For laughter, pick the 'tail end' — after the peak — where eyes are open and joy is still visible.\n"
        "\n"
        "EXPRESSION QUALITY (ranked by preference):\n"
        "1. Duchenne smile — genuine smile that reaches the eyes (crow's feet, brightened gaze).\n"
        "2. Authentic laughter or strong reaction — relaxed jaw, real emotion.\n"
        "3. Quiet connection — contemplation, awe, or a soft meaningful look.\n"
        "4. Neutral with strong eye contact — confident and readable.\n"
        "Reject forced or 'canned' smiles where only the mouth moves but the eyes are flat.\n"
        "\n"
        "SEQUENCING GOAL (think like a storyteller, not just a quality filter):\n"
        "- Select frames that together form a narrative arc.\n"
        "- Avoid picking multiple nearly identical frames — prioritize variety.\n"
        "\n"
        f"There are {len(batch_frames)} frames indexed 0 to {len(batch_frames)-1}.\n"
        f"Pick the best {total_select} (or fewer if there aren't enough good ones).\n"
        'Return ONLY a JSON object exactly like: {"indices": [0, 1, 2]}'
    )

    parts = [prompt]
    for i, (frame, _score, _orig) in enumerate(batch_frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        buf.seek(0)
        parts.append(f"\nFrame {i}:")
        parts.append(Image.open(buf))

    if cancel_event and cancel_event.is_set():
        return []

    for attempt in range(MAX_RETRIES):
        if cancel_event and cancel_event.is_set():
            return []
        try:
            response = model.generate_content(parts)
            text = response.text.strip()
            m = re.search(r'\{[^{}]*"indices"[^{}]*\}', text, re.DOTALL)
            if m:
                raw = json.loads(m.group())["indices"]
            else:
                raw = list(map(int, re.findall(r'\b\d+\b', text)))
            return [i for i in raw if isinstance(i, int) and 0 <= i < len(batch_frames)]
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise exc
    return []


def gemini_select(top_frames, api_key: str, n_select: int = GEMINI_SELECTS,
                  progress_cb=None, cancel_event: threading.Event = None) -> list[int]:
    genai.configure(api_key=api_key)
    model   = genai.GenerativeModel(GEMINI_MODEL)
    n       = len(top_frames)
    batches = [top_frames[i:i + GEMINI_BATCH] for i in range(0, n, GEMINI_BATCH)]
    per_batch_select = max(3, n_select // len(batches) + 1)

    finalists = []
    for b_idx, batch in enumerate(batches):
        if cancel_event and cancel_event.is_set():
            return []
        if progress_cb:
            progress_cb(55 + b_idx / len(batches) * 20)
        local_picks   = _gemini_pick_batch(model, batch, b_idx * GEMINI_BATCH,
                                           per_batch_select, cancel_event)
        global_offset = b_idx * GEMINI_BATCH
        for li in local_picks:
            gi = global_offset + li
            finalists.append((top_frames[gi][0], top_frames[gi][1],
                              top_frames[gi][2], gi))

    if not finalists or (cancel_event and cancel_event.is_set()):
        return []

    if progress_cb:
        progress_cb(78)

    if len(finalists) <= n_select:
        return [f[3] for f in finalists]

    final_batch  = [(f[0], f[1], f[2]) for f in finalists]
    final_picks  = _gemini_pick_batch(model, final_batch, 0,
                                      n_select, cancel_event)
    if progress_cb:
        progress_cb(90)
    return [finalists[i][3] for i in final_picks[:n_select]]


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
        self.title("Frame Extractor")
        self.configure(bg=BG)
        self.minsize(900, 720)
        self._tk_images:  list[ImageTk.PhotoImage] = []
        self._frame_data: list                     = []
        self._cancel_event  = threading.Event()
        self._last_save_dir = os.path.expanduser("~/Pictures")
        self._build_ui()

    # ── build ─────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_header()
        self._build_how_it_works()
        self._build_controls()
        self._build_progress()
        self._separator()
        self._build_results_header()
        self._build_grid()

    def _build_header(self):
        hdr = tk.Frame(self, bg=BG, pady=22)
        hdr.pack(fill="x", padx=32)

        tk.Label(
            hdr, text="Frame Extractor",
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
            fg=MUTED, bg=BG, anchor="w", letterSpacing=2,
        ).pack(anchor="w", pady=(0, 6))

        # two-column layout: Stage 1 | Stage 2
        cols = tk.Frame(outer, bg=BG)
        cols.pack(fill="x")
        cols.columnconfigure(0, weight=1)
        cols.columnconfigure(1, weight=1)

        self._info_card(
            cols, col=0,
            number="01",
            title="OpenCV sharpness pass",
            body=(
                "Samples 1 frame every N frames. Each sample is scored with "
                "Laplacian-variance — a fast, accurate measure of edge sharpness. "
                "A temporal-diversity filter then prunes near-duplicate frames so "
                "candidates are spread across the full timeline."
            ),
        )
        self._info_card(
            cols, col=1,
            number="02",
            title="Gemini visual selection",
            body=(
                "The sharpest candidates are sent to Gemini in batches. Gemini "
                "evaluates them as a professional photo editor — prioritising open eyes, "
                "genuine Duchenne smiles, flattering angles, and story arc — then "
                "returns the 10 best frames."
            ),
        )

        self._separator(pady=(16, 0))

    def _info_card(self, parent, col, number, title, body):
        card = tk.Frame(parent, bg=BG, padx=(0 if col == 0 else 24), pady=0)
        card.grid(row=0, column=col, sticky="nsew")

        tk.Label(
            card, text=number,
            font=("Helvetica", 28, "bold"),
            fg=BORDER, bg=BG, anchor="w",
        ).pack(anchor="w")

        tk.Label(
            card, text=title,
            font=("Helvetica", 11, "bold"),
            fg=WHITE, bg=BG, anchor="w",
        ).pack(anchor="w", pady=(0, 5))

        tk.Label(
            card, text=body,
            font=("Helvetica", 9),
            fg=TEXT, bg=BG, anchor="w",
            justify="left", wraplength=360,
        ).pack(anchor="w")

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

        # options row
        opt = tk.Frame(ctrl, bg=BG)
        opt.pack(anchor="w", pady=(0, 12))

        self.sample_var  = tk.IntVar(value=SAMPLE_EVERY_N)
        self.top_var     = tk.IntVar(value=TOP_N_SHARP)
        self.gap_var     = tk.IntVar(value=TEMPORAL_GAP)
        self.selects_var = tk.IntVar(value=GEMINI_SELECTS)

        self._spin(opt, "SAMPLE EVERY",  self.sample_var,  1,    120)
        self._spin(opt, "TOP SHARP",     self.top_var,     5,    300)
        self._spin(opt, "MIN FRAME GAP", self.gap_var,     0,   9999)
        self._spin(opt, "FRAMES TO PICK", self.selects_var, 1,     50)

        # buttons
        btn_row = tk.Frame(ctrl, bg=BG)
        btn_row.pack(anchor="w")

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

        tk.Label(wrap, text=label,
                 font=("Helvetica", 7, "bold"),
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

    def _spin(self, parent, label, var, lo, hi):
        grp = tk.Frame(parent, bg=BG, padx=(0, 20))
        grp.pack(side="left")
        tk.Label(grp, text=label,
                 font=("Helvetica", 7, "bold"),
                 fg=MUTED, bg=BG).pack(anchor="w", pady=(0, 3))
        tk.Spinbox(
            grp, from_=lo, to=hi, textvariable=var,
            width=6,
            bg=BG2, fg=WHITE, insertbackground=WHITE,
            relief="flat", bd=0,
            buttonbackground=BG2, buttonforeground=MUTED,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=WHITE,
            font=("Helvetica", 10),
        ).pack()

    def _build_progress(self):
        prog = tk.Frame(self, bg=BG, padx=32)
        prog.pack(fill="x", pady=(0, 4))

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(prog, textvariable=self.status_var,
                 fg=MUTED, bg=BG,
                 font=("Helvetica", 8),
                 anchor="w").pack(anchor="w", pady=(0, 4))

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("W.TProgressbar",
                        troughcolor=BG2,
                        background=WHITE,
                        bordercolor=BG,
                        lightcolor=WHITE,
                        darkcolor=WHITE,
                        thickness=2)
        self.progress = ttk.Progressbar(prog, length=800,
                                        mode="determinate",
                                        style="W.TProgressbar")
        self.progress.pack(fill="x", pady=(0, 8))

    def _build_results_header(self):
        row = tk.Frame(self, bg=BG, padx=32, pady=(10, 4))
        row.pack(fill="x")

        tk.Label(row, text="RESULTS",
                 font=("Helvetica", 7, "bold"),
                 fg=MUTED, bg=BG).pack(side="left")

        tk.Label(row,
                 text="click any frame to zoom  ·  esc to close",
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

    def _build_grid(self):
        wrapper = tk.Frame(self, bg=BG)
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

        self._cells:    list[tk.Label]  = []
        self._captions: list[tk.Label]  = []
        self._dl_btns:  list[tk.Button] = []

        COLS = 2
        for i in range(DISPLAY_N):
            r, c = divmod(i, COLS)
            self._grid_frame.columnconfigure(c, weight=1)

            # card: 1px border via outer/inner frame trick
            outer = tk.Frame(self._grid_frame, bg=BORDER)
            outer.grid(row=r, column=c, padx=6, pady=6, sticky="nsew")

            inner = tk.Frame(outer, bg=BG2)
            inner.pack(fill="both", expand=True, padx=1, pady=1)

            img_lbl = tk.Label(inner, bg=BG2, text="",
                               relief="flat", cursor="hand2")
            img_lbl.pack(fill="both", expand=True)
            img_lbl.bind("<Button-1>", lambda e, idx=i: self._zoom(idx))

            footer = tk.Frame(inner, bg=BG2, pady=5)
            footer.pack(fill="x", padx=8)

            cap_lbl = tk.Label(footer, bg=BG2, fg=MUTED,
                               text=f"—",
                               font=("Helvetica", 7), anchor="w")
            cap_lbl.pack(side="left")

            dl_btn = tk.Button(
                footer, text="Save",
                command=lambda idx=i: self._download_single(idx),
                bg=BG2, fg=MUTED,
                font=("Helvetica", 7), relief="flat",
                padx=6, pady=0, cursor="hand2",
                state="disabled",
                activebackground=BG2, activeforeground=WHITE,
            )
            dl_btn.pack(side="right")

            self._cells.append(img_lbl)
            self._captions.append(cap_lbl)
            self._dl_btns.append(dl_btn)

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

        self._cancel_event.clear()
        self.run_btn.config(state="disabled")
        self.cancel_btn.config(state="normal", fg=TEXT)
        self.dl_all_btn.config(state="disabled")
        self._clear_grid()
        threading.Thread(target=self._pipeline, args=(video, key), daemon=True).start()

    def _pipeline(self, video, key):
        try:
            sample_every = self.sample_var.get()
            top_n        = self.top_var.get()
            min_gap      = self.gap_var.get()
            n_select     = max(1, self.selects_var.get())

            self._update("01 / scanning with OpenCV…", 0)
            raw = extract_sharp_frames(
                video, sample_every,
                progress_cb=lambda v: self._update(None, v),
                cancel_event=self._cancel_event,
            )

            if self._cancel_event.is_set():
                self._update("Cancelled.", 0)
                return

            diverse = temporal_diversity_filter(raw, min_gap, top_n)
            self._update(
                f"01 / done — {len(diverse)} candidates  ·  "
                "02 / asking Gemini…", 50
            )

            picked = gemini_select(
                diverse, key, n_select=n_select,
                progress_cb=lambda v: self._update(None, v),
                cancel_event=self._cancel_event,
            )

            if self._cancel_event.is_set():
                self._update("Cancelled.", 0)
                return

            selected = [diverse[i] for i in picked[:n_select]]
            self._update(f"{len(selected)} frames selected.", 100)
            self.after(0, lambda: self._show_frames(selected))

        except Exception as exc:
            self.after(0, lambda e=exc: messagebox.showerror("Error", str(e)))
            self._update(f"Error: {exc}", 0)
        finally:
            self.after(0, lambda: self.run_btn.config(state="normal"))
            self.after(0, lambda: self.cancel_btn.config(state="disabled", fg=MUTED))

    def _zoom(self, idx: int):
        if idx >= len(self._frame_data):
            return
        frame, score, orig_idx = self._frame_data[idx]
        ZoomWindow(self, frame,
                   title=f"#{idx+1}  frame {orig_idx}  ·  sharpness {score:,.0f}")

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
        frame, score, orig_idx = self._frame_data[idx]
        path = self._save_dialog(f"frame_{idx+1}_{orig_idx}.jpg")
        if path:
            self._write_frame(path, frame)
            self._update(f"Saved → {os.path.basename(path)}", self.progress["value"])

    def _download_all(self):
        if not self._frame_data:
            return
        folder = filedialog.askdirectory(
            title="Save all frames to…",
            initialdir=self._last_save_dir,
        )
        if not folder:
            return
        self._last_save_dir = folder
        for i, (frame, _score, orig_idx) in enumerate(self._frame_data):
            self._write_frame(
                os.path.join(folder, f"frame_{i+1}_{orig_idx}.jpg"), frame
            )
        self._update(
            f"Saved {len(self._frame_data)} frames → {folder}",
            self.progress["value"],
        )

    # ── helpers ───────────────────────────────────────────────────────────────
    def _update(self, msg, value):
        if msg is not None:
            self.after(0, lambda m=msg: self.status_var.set(m))
        self.after(0, lambda v=value: self.progress.config(value=v))

    def _clear_grid(self):
        self._tk_images.clear()
        self._frame_data.clear()
        for lbl, cap, btn in zip(self._cells, self._captions, self._dl_btns):
            lbl.config(image="", text="")
            cap.config(text="—")
            btn.config(state="disabled", fg=MUTED)

    def _show_frames(self, frames_data):
        self._tk_images.clear()
        self._frame_data = list(frames_data)

        for i, (lbl, cap_lbl, dl_btn) in enumerate(
            zip(self._cells, self._captions, self._dl_btns)
        ):
            if i >= len(frames_data):
                lbl.config(image="", text="")
                cap_lbl.config(text="—")
                dl_btn.config(state="disabled", fg=MUTED)
                continue

            frame, score, orig_idx = frames_data[i]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            cell_w = max(lbl.winfo_width(),  380)
            cell_h = max(lbl.winfo_height(), 260)
            img.thumbnail((cell_w - 2, cell_h - 2), Image.LANCZOS)

            tk_img = ImageTk.PhotoImage(img)
            self._tk_images.append(tk_img)

            lbl.config(image=tk_img, text="")
            cap_lbl.config(
                text=f"#{i+1}  ·  frame {orig_idx}  ·  ♯ {score:,.0f}",
                fg=TEXT,
            )
            dl_btn.config(state="normal", fg=TEXT)

        self.dl_all_btn.config(state="normal", fg=TEXT)


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()
