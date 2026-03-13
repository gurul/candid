#!/usr/bin/env python3
"""
Video Frame Extractor — two-stage pipeline
  Stage 1 : OpenCV Laplacian-variance sharpness filter
  Stage 2 : Gemini visual selection
  UI      : Tkinter 2×2 grid showing the 4 best frames
"""

import cv2
import io
import json
import os
import re
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import google.generativeai as genai
from PIL import Image, ImageTk

# ── Config ──────────────────────────────────────────────────────────────────
SAMPLE_EVERY_N = 30          # sample 1 frame every N frames
TOP_N_SHARP    = 150         # keep top-N by Laplacian score before Gemini
GEMINI_SELECTS = 5           # how many frames Gemini picks
DISPLAY_N      = 4           # frames shown in the UI grid
THUMB_W, THUMB_H = 320, 240  # resize before sending to Gemini (saves tokens)
GEMINI_MODEL   = "gemini-2.0-flash"   # change to gemini-2.5-flash-preview etc.
OUTPUT_DIR     = "output"
# ────────────────────────────────────────────────────────────────────────────


# ── Stage 1 : sharpness scoring ─────────────────────────────────────────────
def extract_sharp_frames(video_path: str, progress_cb=None):
    """Sample every Nth frame, score with Laplacian variance, return top 150."""
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames, scores, orig_indices = [], [], []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % SAMPLE_EVERY_N == 0:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            frames.append(frame.copy())
            scores.append(score)
            orig_indices.append(idx)
            if progress_cb:
                progress_cb(min(idx / max(total, 1) * 48, 48))
        idx += 1

    cap.release()

    order = np.argsort(scores)[::-1][:TOP_N_SHARP]
    return [(frames[i], scores[i], orig_indices[i]) for i in order]


# ── Stage 2 : Gemini selection ───────────────────────────────────────────────
def gemini_select(top_frames, api_key: str, progress_cb=None):
    """Send all top frames to Gemini in one request; return chosen indices."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    prompt = (
        "You are a professional photo editor reviewing video frame candidates.\n"
        "From these candidate frames, select the 5 best based on composition, "
        "lighting, and subject clarity.\n"
        f"There are {len(top_frames)} frames indexed 0 to {len(top_frames)-1}.\n"
        'Return ONLY a JSON object exactly like: {"indices": [0, 1, 2, 3, 4]}'
    )

    parts = [prompt]
    for i, (frame, _score, _orig) in enumerate(top_frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        buf.seek(0)
        pil_img = Image.open(buf)
        parts.append(f"\nFrame {i}:")
        parts.append(pil_img)

    if progress_cb:
        progress_cb(55)

    response = model.generate_content(parts)

    if progress_cb:
        progress_cb(85)

    text = response.text.strip()
    # extract JSON
    m = re.search(r'\{[^{}]*"indices"[^{}]*\}', text, re.DOTALL)
    if m:
        raw = json.loads(m.group())["indices"]
    else:
        raw = list(map(int, re.findall(r'\b\d+\b', text)))

    # validate bounds
    picked = [i for i in raw if isinstance(i, int) and 0 <= i < len(top_frames)]
    return picked[:GEMINI_SELECTS]


# ── Save ─────────────────────────────────────────────────────────────────────
def save_frames(frames_data, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for rank, (frame, score, orig_idx) in enumerate(frames_data, 1):
        path = os.path.join(output_dir, f"best_{rank}_frame{orig_idx}.jpg")
        cv2.imwrite(path, frame)
        paths.append(path)
    return paths


# ── UI ───────────────────────────────────────────────────────────────────────
BG       = "#0d1117"
BG2      = "#161b22"
ACCENT   = "#58a6ff"
DANGER   = "#f85149"
TEXT     = "#c9d1d9"
MUTED    = "#6e7681"
SUCCESS  = "#3fb950"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video Frame Extractor")
        self.configure(bg=BG)
        self.minsize(760, 620)
        self._tk_images: list[ImageTk.PhotoImage] = []
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
        tk.Label(opt, text="sharpest", fg=MUTED, bg=BG,
                 font=("Helvetica", 9)).pack(side="left")

        # run button
        self.run_btn = tk.Button(
            ctrl, text="▶  Extract Best Frames",
            command=self._run,
            bg=ACCENT, fg=BG,
            font=("Helvetica", 11, "bold"),
            relief="flat", padx=18, pady=7, cursor="hand2",
        )
        self.run_btn.pack(pady=(10, 4))

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

        # frame grid label
        tk.Label(self, text="Best frames selected by Gemini",
                 fg=MUTED, bg=BG, font=("Helvetica", 9, "italic")).pack(pady=(8, 0))

        # 2×2 grid
        grid = tk.Frame(self, bg=BG, padx=16, pady=8)
        grid.pack(fill="both", expand=True)
        self._cells: list[tk.Label] = []
        self._captions: list[tk.Label] = []
        for r in range(2):
            grid.rowconfigure(r, weight=1)
            for c in range(2):
                grid.columnconfigure(c, weight=1)
                outer = tk.Frame(grid, bg=BG2, padx=2, pady=2)
                outer.grid(row=r, column=c, padx=6, pady=6, sticky="nsew")

                img_lbl = tk.Label(outer, bg=BG2, text="",
                                   relief="flat")
                img_lbl.pack(fill="both", expand=True)

                cap_lbl = tk.Label(outer, bg=BG2, fg=MUTED,
                                   text=f"Slot {r*2+c+1}",
                                   font=("Helvetica", 8))
                cap_lbl.pack()

                self._cells.append(img_lbl)
                self._captions.append(cap_lbl)

        # footer
        self.save_var = tk.StringVar()
        tk.Label(self, textvariable=self.save_var,
                 fg=SUCCESS, bg=BG, font=("Helvetica", 9)).pack(pady=(2, 10))

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

        self.run_btn.config(state="disabled")
        self.save_var.set("")
        self._clear_grid()
        threading.Thread(target=self._pipeline, args=(video, key), daemon=True).start()

    def _pipeline(self, video, key):
        try:
            n = self.sample_var.get()
            top_n = self.top_var.get()

            self._update("Stage 1 — scanning frames with OpenCV…", 0)
            top = extract_sharp_frames(
                video,
                progress_cb=lambda v: self._update(None, v),
            )
            # respect UI spinbox value
            top = top[:top_n]

            self._update(
                f"Stage 1 done — kept {len(top)} sharp frames.  "
                "Sending to Gemini…", 50
            )
            picked = gemini_select(
                top, key,
                progress_cb=lambda v: self._update(None, v),
            )

            self._update("Saving frames to disk…", 92)
            selected = [top[i] for i in picked[:GEMINI_SELECTS]]
            paths    = save_frames(selected)

            self._update("Done ✓", 100)
            self.after(0, lambda: self._show_frames(selected, paths))

        except Exception as exc:
            self.after(0, lambda e=exc: messagebox.showerror("Pipeline error", str(e)))
            self._update(f"Error: {exc}", 0)
        finally:
            self.after(0, lambda: self.run_btn.config(state="normal"))

    # ── helpers ───────────────────────────────────────────────────────────────
    def _update(self, msg, value):
        if msg is not None:
            self.after(0, lambda m=msg: self.status_var.set(m))
        self.after(0, lambda v=value: self.progress.config(value=v))

    def _clear_grid(self):
        self._tk_images.clear()
        for lbl, cap in zip(self._cells, self._captions):
            lbl.config(image="", text="")
            cap.config(text="")

    def _show_frames(self, frames_data, paths):
        self._tk_images.clear()
        for i, (lbl, cap_lbl) in enumerate(zip(self._cells, self._captions)):
            if i >= len(frames_data):
                lbl.config(image="", text="")
                cap_lbl.config(text="")
                continue

            frame, score, orig_idx = frames_data[i]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            # fit inside the cell dynamically
            cell_w = max(lbl.winfo_width(),  340)
            cell_h = max(lbl.winfo_height(), 240)
            img.thumbnail((cell_w - 4, cell_h - 4), Image.LANCZOS)

            tk_img = ImageTk.PhotoImage(img)
            self._tk_images.append(tk_img)

            lbl.config(image=tk_img, text="")
            cap_lbl.config(
                text=f"Rank {i+1}  |  frame #{orig_idx}  |  sharpness {score:,.0f}",
                fg=TEXT,
            )

        abs_out = os.path.abspath(OUTPUT_DIR)
        self.save_var.set(f"Saved {len(frames_data)} frame(s) → {abs_out}")


# ── entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()
