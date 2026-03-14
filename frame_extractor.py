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
SAMPLE_EVERY_N   = 30          # sample 1 frame every N frames
TOP_N_SHARP      = 150         # keep top-N by Laplacian score before Gemini
GEMINI_SELECTS   = 10          # how many frames Gemini picks
DISPLAY_N        = 10          # frames shown in the UI grid
THUMB_W, THUMB_H = 320, 240   # resize before sending to Gemini (saves tokens)
GEMINI_MODEL     = "gemini-2.5-flash"
GEMINI_BATCH     = 30          # frames per Gemini sub-batch
TEMPORAL_GAP     = 90          # min frame gap for diversity filter (0 = off)
MAX_RETRIES      = 3           # Gemini API retry attempts
RETRY_DELAY      = 2.0         # seconds between retries
# ────────────────────────────────────────────────────────────────────────────


# ── Stage 1 : sharpness scoring ─────────────────────────────────────────────
def extract_sharp_frames(video_path: str, sample_every: int, progress_cb=None,
                         cancel_event: threading.Event = None):
    """Sample every Nth frame, score with Laplacian variance, return top frames."""
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
    """Return fps, frame_count, duration_s, width, height."""
    cap = cv2.VideoCapture(video_path)
    fps     = cap.get(cv2.CAP_PROP_FPS) or 1
    frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": fps,
        "frames": frames,
        "duration_s": frames / fps,
        "width": width,
        "height": height,
    }


def temporal_diversity_filter(frames, min_gap: int, limit: int):
    """
    From a score-sorted list, keep frames that are at least min_gap apart
    in the original video, up to `limit` frames.
    """
    if min_gap <= 0:
        return frames[:limit]
    kept = []
    last_orig = -min_gap - 1
    for item in frames:
        _frame, _score, orig_idx = item
        if orig_idx - last_orig >= min_gap:
            kept.append(item)
            last_orig = orig_idx
            if len(kept) >= limit:
                break
    return kept


# ── Stage 2 : Gemini selection ───────────────────────────────────────────────
def _gemini_pick_batch(model, batch_frames, batch_offset: int,
                       total_select: int, cancel_event) -> list[int]:
    """Ask Gemini to pick up to total_select from one batch; returns local indices."""
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


def gemini_select(top_frames, api_key: str, progress_cb=None,
                  cancel_event: threading.Event = None) -> list[int]:
    """
    Send frames to Gemini in batches of GEMINI_BATCH.
    Each batch returns its local best; combine and do a final selection pass.
    Returns indices into top_frames.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    n = len(top_frames)
    batches = [top_frames[i:i + GEMINI_BATCH] for i in range(0, n, GEMINI_BATCH)]
    per_batch_select = max(3, GEMINI_SELECTS // len(batches) + 1)

    finalists = []   # (frame, score, orig_idx, global_idx)
    for b_idx, batch in enumerate(batches):
        if cancel_event and cancel_event.is_set():
            return []
        if progress_cb:
            progress_cb(55 + b_idx / len(batches) * 20)
        local_picks = _gemini_pick_batch(model, batch, b_idx * GEMINI_BATCH,
                                         per_batch_select, cancel_event)
        global_offset = b_idx * GEMINI_BATCH
        for li in local_picks:
            gi = global_offset + li
            finalists.append((top_frames[gi][0], top_frames[gi][1],
                              top_frames[gi][2], gi))

    if not finalists or cancel_event and cancel_event.is_set():
        return []

    if progress_cb:
        progress_cb(78)

    # Final selection pass if we have more finalists than needed
    if len(finalists) <= GEMINI_SELECTS:
        return [f[3] for f in finalists]

    # Build a mini-batch from finalists for final ranking
    final_batch = [(f[0], f[1], f[2]) for f in finalists]
    final_picks = _gemini_pick_batch(model, final_batch, 0,
                                     GEMINI_SELECTS, cancel_event)

    if progress_cb:
        progress_cb(90)

    return [finalists[i][3] for i in final_picks[:GEMINI_SELECTS]]


# ── UI ───────────────────────────────────────────────────────────────────────
BG      = "#0d1117"
BG2     = "#161b22"
ACCENT  = "#58a6ff"
DANGER  = "#f85149"
TEXT    = "#c9d1d9"
MUTED   = "#6e7681"
SUCCESS = "#3fb950"


class ZoomWindow(tk.Toplevel):
    """Full-resolution popup on thumbnail click."""
    def __init__(self, parent, frame_bgr, title=""):
        super().__init__(parent)
        self.title(title)
        self.configure(bg=BG)
        self.resizable(True, True)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        img.thumbnail((int(sw * 0.85), int(sh * 0.85)), Image.LANCZOS)

        self._tk_img = ImageTk.PhotoImage(img)
        lbl = tk.Label(self, image=self._tk_img, bg=BG)
        lbl.pack(padx=8, pady=8)

        self.bind("<Escape>", lambda _: self.destroy())
        lbl.bind("<Button-1>", lambda _: self.destroy())


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video Frame Extractor")
        self.configure(bg=BG)
        self.minsize(860, 700)
        self._tk_images: list[ImageTk.PhotoImage] = []
        self._frame_data: list = []          # (bgr_frame, score, orig_idx)
        self._cancel_event = threading.Event()
        self._last_save_dir = os.path.expanduser("~/Pictures")
        self._build_ui()

    # ── layout ───────────────────────────────────────────────────────────────
    def _build_ui(self):
        # header
        hdr = tk.Frame(self, bg=BG2, pady=14)
        hdr.pack(fill="x")
        tk.Label(
            hdr, text="  Video Frame Extractor",
            font=("Helvetica", 17, "bold"),
            fg=ACCENT, bg=BG2,
        ).pack(side="left", padx=20)
        tk.Label(
            hdr, text=f"model: {GEMINI_MODEL}",
            font=("Helvetica", 9), fg=MUTED, bg=BG2,
        ).pack(side="right", padx=20)

        # controls panel
        ctrl = tk.Frame(self, bg=BG, padx=20, pady=12)
        ctrl.pack(fill="x")

        self.video_var = tk.StringVar()
        self.key_var   = tk.StringVar(value=os.environ.get("GEMINI_API_KEY", ""))

        self._ctrl_row(ctrl, "Video file", self.video_var,
                       browse_cmd=self._browse_video)
        self._ctrl_row(ctrl, "Gemini API key", self.key_var, secret=True)

        # video metadata label
        self.meta_var = tk.StringVar(value="")
        tk.Label(ctrl, textvariable=self.meta_var,
                 fg=MUTED, bg=BG, font=("Helvetica", 8, "italic")).pack(anchor="w")

        # options row
        opt = tk.Frame(ctrl, bg=BG)
        opt.pack(fill="x", pady=4)
        tk.Label(opt, text="Sample every", fg=MUTED, bg=BG,
                 font=("Helvetica", 9)).pack(side="left")
        self.sample_var = tk.IntVar(value=SAMPLE_EVERY_N)
        tk.Spinbox(opt, from_=1, to=120, textvariable=self.sample_var,
                   width=4, bg=BG2, fg=TEXT, relief="flat",
                   buttonbackground=BG2).pack(side="left", padx=4)
        tk.Label(opt, text="frames   Keep top", fg=MUTED, bg=BG,
                 font=("Helvetica", 9)).pack(side="left")
        self.top_var = tk.IntVar(value=TOP_N_SHARP)
        tk.Spinbox(opt, from_=5, to=300, textvariable=self.top_var,
                   width=4, bg=BG2, fg=TEXT, relief="flat",
                   buttonbackground=BG2).pack(side="left", padx=4)
        tk.Label(opt, text="sharpest   Min frame gap", fg=MUTED, bg=BG,
                 font=("Helvetica", 9)).pack(side="left")
        self.gap_var = tk.IntVar(value=TEMPORAL_GAP)
        tk.Spinbox(opt, from_=0, to=9999, textvariable=self.gap_var,
                   width=5, bg=BG2, fg=TEXT, relief="flat",
                   buttonbackground=BG2).pack(side="left", padx=4)
        tk.Label(opt, text="(0=off)", fg=MUTED, bg=BG,
                 font=("Helvetica", 9)).pack(side="left")

        # run / cancel buttons
        btn_row = tk.Frame(ctrl, bg=BG)
        btn_row.pack(pady=(10, 4))

        self.run_btn = tk.Button(
            btn_row, text="▶  Extract Best Frames",
            command=self._run,
            bg=ACCENT, fg=BG,
            font=("Helvetica", 11, "bold"),
            relief="flat", padx=18, pady=7, cursor="hand2",
        )
        self.run_btn.pack(side="left", padx=(0, 8))

        self.cancel_btn = tk.Button(
            btn_row, text="✕  Cancel",
            command=self._cancel,
            bg=BG2, fg=DANGER,
            font=("Helvetica", 11, "bold"),
            relief="flat", padx=12, pady=7, cursor="hand2",
            state="disabled",
        )
        self.cancel_btn.pack(side="left")

        # progress
        self.status_var = tk.StringVar(value="Ready — select a video to begin")
        tk.Label(self, textvariable=self.status_var,
                 fg=MUTED, bg=BG, font=("Helvetica", 9)).pack()

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TProgressbar", troughcolor=BG2,
                        background=ACCENT, thickness=6)
        self.progress = ttk.Progressbar(self, length=720,
                                        mode="determinate", style="TProgressbar")
        self.progress.pack(pady=(2, 10), padx=20, fill="x")

        # divider
        tk.Frame(self, bg=BG2, height=1).pack(fill="x")

        # section label + download-all button on same row
        lbl_row = tk.Frame(self, bg=BG)
        lbl_row.pack(fill="x", padx=20, pady=(8, 0))
        tk.Label(lbl_row, text="Best frames selected by Gemini  —  click any image to zoom",
                 fg=MUTED, bg=BG, font=("Helvetica", 9, "italic")).pack(side="left")
        self.dl_all_btn = tk.Button(
            lbl_row, text="⬇  Download All",
            command=self._download_all,
            bg=BG2, fg=ACCENT,
            font=("Helvetica", 9, "bold"),
            relief="flat", padx=10, pady=3, cursor="hand2",
            state="disabled",
        )
        self.dl_all_btn.pack(side="right")

        # ── scrollable 2-column grid ─────────────────────────────────────────
        wrapper = tk.Frame(self, bg=BG)
        wrapper.pack(fill="both", expand=True, padx=12, pady=(6, 12))

        canvas = tk.Canvas(wrapper, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(wrapper, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self._grid_frame = tk.Frame(canvas, bg=BG)
        self._canvas_window = canvas.create_window(
            (0, 0), window=self._grid_frame, anchor="nw"
        )

        def _on_frame_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(e):
            canvas.itemconfig(self._canvas_window, width=e.width)

        self._grid_frame.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        # platform-aware mouse-wheel scrolling
        if platform.system() == "Darwin":
            canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1 * e.delta, "units"))
        else:
            canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
            canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
            canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        # pre-build 10 cells (2 columns × 5 rows)
        self._cells: list[tk.Label]    = []
        self._captions: list[tk.Label] = []
        self._dl_btns: list[tk.Button] = []

        COLS = 2
        for i in range(DISPLAY_N):
            r, c = divmod(i, COLS)
            self._grid_frame.columnconfigure(c, weight=1)

            outer = tk.Frame(self._grid_frame, bg=BG2, padx=2, pady=2)
            outer.grid(row=r, column=c, padx=6, pady=6, sticky="nsew")

            img_lbl = tk.Label(outer, bg=BG2, text="",
                               relief="flat", cursor="hand2")
            img_lbl.pack(fill="both", expand=True)
            img_lbl.bind("<Button-1>", lambda e, idx=i: self._zoom(idx))

            bottom = tk.Frame(outer, bg=BG2)
            bottom.pack(fill="x")

            cap_lbl = tk.Label(bottom, bg=BG2, fg=MUTED,
                               text=f"Slot {i+1}",
                               font=("Helvetica", 8), anchor="w")
            cap_lbl.pack(side="left", padx=4, pady=2)

            dl_btn = tk.Button(
                bottom, text="⬇ Save",
                command=lambda idx=i: self._download_single(idx),
                bg=BG, fg=ACCENT,
                font=("Helvetica", 8), relief="flat",
                padx=6, pady=1, cursor="hand2",
                state="disabled",
            )
            dl_btn.pack(side="right", padx=4, pady=2)

            self._cells.append(img_lbl)
            self._captions.append(cap_lbl)
            self._dl_btns.append(dl_btn)

    def _ctrl_row(self, parent, label, var, browse_cmd=None, secret=False):
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", pady=3)
        tk.Label(row, text=label + ":", width=13, anchor="w",
                 fg=MUTED, bg=BG, font=("Helvetica", 9)).pack(side="left")
        kw = dict(show="•") if secret else {}
        entry = tk.Entry(row, textvariable=var, width=52,
                         bg=BG2, fg=TEXT, insertbackground=TEXT,
                         relief="flat", bd=6, font=("Helvetica", 10), **kw)
        entry.pack(side="left", padx=4)
        if browse_cmd:
            tk.Button(row, text="Browse…", command=browse_cmd,
                      bg=BG2, fg=ACCENT, relief="flat",
                      padx=8, cursor="hand2").pack(side="left")

    # ── actions ───────────────────────────────────────────────────────────────
    def _browse_video(self):
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
                       ("All files", "*.*")]
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
                f"{m['width']}×{m['height']}  •  {m['fps']:.2f} fps  "
                f"•  {mins}m {secs:02d}s  •  {m['frames']:,} frames"
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
        self.cancel_btn.config(state="normal")
        self.dl_all_btn.config(state="disabled")
        self._clear_grid()
        threading.Thread(target=self._pipeline, args=(video, key), daemon=True).start()

    def _pipeline(self, video, key):
        try:
            sample_every = self.sample_var.get()
            top_n        = self.top_var.get()
            min_gap      = self.gap_var.get()

            self._update("Stage 1 — scanning frames with OpenCV…", 0)
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
                f"Stage 1 done — {len(diverse)} diverse sharp frames.  "
                "Sending to Gemini…", 50
            )

            picked = gemini_select(
                diverse, key,
                progress_cb=lambda v: self._update(None, v),
                cancel_event=self._cancel_event,
            )

            if self._cancel_event.is_set():
                self._update("Cancelled.", 0)
                return

            selected = [diverse[i] for i in picked[:GEMINI_SELECTS]]
            self._update(f"Done ✓  —  {len(selected)} best frames selected", 100)
            self.after(0, lambda: self._show_frames(selected))

        except Exception as exc:
            self.after(0, lambda e=exc: messagebox.showerror("Pipeline error", str(e)))
            self._update(f"Error: {exc}", 0)
        finally:
            self.after(0, lambda: self.run_btn.config(state="normal"))
            self.after(0, lambda: self.cancel_btn.config(state="disabled"))

    # ── zoom ──────────────────────────────────────────────────────────────────
    def _zoom(self, idx: int):
        if idx >= len(self._frame_data):
            return
        frame, score, orig_idx = self._frame_data[idx]
        ZoomWindow(self, frame, title=f"Rank {idx+1}  |  frame #{orig_idx}  |  sharpness {score:,.0f}")

    # ── download helpers ──────────────────────────────────────────────────────
    def _save_dialog(self, initial_file: str):
        """Open save dialog, remember the chosen directory."""
        path = filedialog.asksaveasfilename(
            title="Save frame",
            initialdir=self._last_save_dir,
            initialfile=initial_file,
            defaultextension=".jpg",
            filetypes=[
                ("JPEG — high quality (95)", "*.jpg"),
                ("JPEG — standard (80)",     "*.jpg"),
                ("PNG — lossless",           "*.png"),
                ("All files",                "*.*"),
            ],
        )
        if path:
            self._last_save_dir = os.path.dirname(path)
        return path

    def _write_frame(self, path: str, frame_bgr):
        """Write frame with quality appropriate to chosen extension."""
        ext = os.path.splitext(path)[1].lower()
        if ext in (".jpg", ".jpeg"):
            # infer quality from filetypes selection — fallback to 95
            cv2.imwrite(path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(path, frame_bgr)

    def _download_single(self, idx: int):
        if idx >= len(self._frame_data):
            return
        frame, score, orig_idx = self._frame_data[idx]
        path = self._save_dialog(f"best_{idx+1}_frame{orig_idx}.jpg")
        if path:
            self._write_frame(path, frame)
            self._update(f"Saved → {path}", self.progress["value"])

    def _download_all(self):
        if not self._frame_data:
            return
        folder = filedialog.askdirectory(
            title="Choose folder to save all frames",
            initialdir=self._last_save_dir,
        )
        if not folder:
            return
        self._last_save_dir = folder
        saved = 0
        for i, (frame, score, orig_idx) in enumerate(self._frame_data):
            path = os.path.join(folder, f"best_{i+1}_frame{orig_idx}.jpg")
            self._write_frame(path, frame)
            saved += 1
        self._update(f"Saved {saved} frame(s) → {folder}", self.progress["value"])

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
            cap.config(text="")
            btn.config(state="disabled")

    def _show_frames(self, frames_data):
        self._tk_images.clear()
        self._frame_data = list(frames_data)

        for i, (lbl, cap_lbl, dl_btn) in enumerate(
            zip(self._cells, self._captions, self._dl_btns)
        ):
            if i >= len(frames_data):
                lbl.config(image="", text="")
                cap_lbl.config(text="")
                dl_btn.config(state="disabled")
                continue

            frame, score, orig_idx = frames_data[i]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            cell_w = max(lbl.winfo_width(),  360)
            cell_h = max(lbl.winfo_height(), 240)
            img.thumbnail((cell_w - 4, cell_h - 4), Image.LANCZOS)

            tk_img = ImageTk.PhotoImage(img)
            self._tk_images.append(tk_img)

            lbl.config(image=tk_img, text="")
            cap_lbl.config(
                text=f"Rank {i+1}  |  frame #{orig_idx}  |  sharpness {score:,.0f}",
                fg=TEXT,
            )
            dl_btn.config(state="normal")

        self.dl_all_btn.config(state="normal")


# ── entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()
