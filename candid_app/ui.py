import argparse
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from .config import HOW_IT_WORKS, PipelineDefaults, Theme
from .models import ExtractionOptions, SelectedFrame
from .pipeline import get_video_metadata, run_pipeline


DEFAULTS = PipelineDefaults()
THEME = Theme()


class ZoomWindow(tk.Toplevel):
    def __init__(self, parent, frame_bgr, title=""):
        super().__init__(parent)
        self.title(title)
        self.configure(bg=THEME.bg)
        self.resizable(True, True)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((int(self.winfo_screenwidth() * 0.86), int(self.winfo_screenheight() * 0.86)), Image.LANCZOS)
        self._img = ImageTk.PhotoImage(img)

        tk.Label(self, image=self._img, bg=THEME.bg).pack(padx=20, pady=20)
        self.bind("<Escape>", lambda _event: self.destroy())


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Candid Frame Extractor")
        self.configure(bg=THEME.bg)
        self.geometry("1180x900")
        self.minsize(1040, 780)

        self._cancel_event = threading.Event()
        self._frame_data: list[SelectedFrame] = []
        self._tk_images: list[ImageTk.PhotoImage] = []
        self._last_video_path = ""
        self._last_save_dir = os.path.expanduser("~/Pictures")

        self.video_var = tk.StringVar()
        self.gemini_key_var = tk.StringVar(value=os.environ.get("GEMINI_API_KEY", ""))
        self.openai_key_var = tk.StringVar(value=os.environ.get("OPENAI_API_KEY", ""))
        self.mode_var = tk.StringVar(value="Smart curation")
        self.meta_var = tk.StringVar(value="No video selected yet.")
        self.status_var = tk.StringVar(value="Ready")
        self.summary_var = tk.StringVar(value="Pick a video, add your API key, and run the scan.")
        self.sample_var = tk.IntVar(value=DEFAULTS.sample_every_n)
        self.top_var = tk.IntVar(value=DEFAULTS.top_n_sharp)
        self.gap_var = tk.IntVar(value=DEFAULTS.temporal_gap)
        self.selects_var = tk.IntVar(value=DEFAULTS.gemini_selects)
        self.q_vars = [tk.StringVar() for _ in range(4)]

        self._configure_styles()
        self._build_ui()

    def _configure_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "Candid.Horizontal.TProgressbar",
            troughcolor=THEME.panel,
            background=THEME.accent,
            bordercolor=THEME.bg,
            lightcolor=THEME.accent,
            darkcolor=THEME.accent,
            thickness=8,
        )

    def _build_ui(self):
        shell = tk.Frame(self, bg=THEME.bg)
        shell.pack(fill="both", expand=True, padx=26, pady=22)

        self._build_hero(shell)

        body = tk.Frame(shell, bg=THEME.bg)
        body.pack(fill="both", expand=True, pady=(18, 0))
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)

        left = tk.Frame(body, bg=THEME.bg)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 18))

        right = tk.Frame(body, bg=THEME.bg)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        self._build_input_panel(left)
        self._build_status_panel(right)
        self._build_results_panel(right)

    def _build_hero(self, parent):
        hero = tk.Frame(parent, bg=THEME.panel, highlightthickness=1, highlightbackground=THEME.border)
        hero.pack(fill="x")

        header = tk.Frame(hero, bg=THEME.panel)
        header.pack(fill="x", padx=24, pady=(22, 10))

        left = tk.Frame(header, bg=THEME.panel)
        left.pack(side="left", fill="both", expand=True)

        tk.Label(
            left,
            text="Candid Frame Extractor",
            bg=THEME.panel,
            fg=THEME.text,
            font=("Avenir Next", 25, "bold"),
        ).pack(anchor="w")
        tk.Label(
            left,
            text="A sharper desktop workflow for pulling polished stills from interviews and reaction videos.",
            bg=THEME.panel,
            fg=THEME.text_muted,
            font=("Avenir Next", 12),
        ).pack(anchor="w", pady=(6, 0))

        pill = tk.Label(
            header,
            text=DEFAULTS.gemini_model,
            bg=THEME.panel_alt,
            fg=THEME.accent_soft,
            font=("Avenir Next", 10, "bold"),
            padx=12,
            pady=7,
        )
        pill.pack(side="right", anchor="n")

        tk.Label(
            hero,
            text=HOW_IT_WORKS,
            wraplength=1020,
            justify="left",
            bg=THEME.panel,
            fg=THEME.text_muted,
            font=("Avenir Next", 11),
        ).pack(fill="x", padx=24, pady=(0, 20))

    def _build_input_panel(self, parent):
        panel = self._card(parent, "Project Setup", "Everything you need to run a clean extraction.")
        panel.pack(fill="x")
        panel.configure(width=360)
        panel.pack_propagate(False)

        self._field(panel, "Video file", self.video_var, browse=self._browse_video, placeholder="Choose a source video")
        self._meta_block(panel)
        self._field(panel, "Gemini API key", self.gemini_key_var, secret=True, placeholder="Required")
        self._field(panel, "OpenAI API key", self.openai_key_var, secret=True, placeholder="Optional unless using question mode")

        mode = tk.Frame(panel, bg=THEME.panel)
        mode.pack(fill="x", pady=(14, 0))
        tk.Label(mode, text="Workflow", bg=THEME.panel, fg=THEME.text_muted, font=("Avenir Next", 10, "bold")).pack(anchor="w")
        self._mode_card(
            mode,
            "Smart curation",
            "Pick the strongest moments from the full video.",
            "Recommended for general use.",
        ).pack(fill="x", pady=(8, 10))
        self._mode_card(
            mode,
            "Question strip",
            "Map four interview questions and pick one frame per answer.",
            "Fill all four prompts and add an OpenAI key to enable this mode.",
        ).pack(fill="x")

        prompts = self._card(panel, "Question Prompts", "Only used in question-strip mode.", inner_pad=False)
        prompts.pack(fill="x", pady=(16, 0))
        for idx in range(4):
            self._field(prompts, f"Q{idx + 1}", self.q_vars[idx], placeholder="Optional unless using question mode")

        settings = self._card(panel, "Tuning", "Control scan density, candidate breadth, and output size.", inner_pad=False)
        settings.pack(fill="x", pady=(16, 0))
        spins = tk.Frame(settings, bg=THEME.panel)
        spins.pack(fill="x", padx=18, pady=(8, 18))
        self._spin(spins, "Sample every", self.sample_var, 1, 120, "frames")
        self._spin(spins, "Top sharp", self.top_var, 5, 300, "candidates")
        self._spin(spins, "Min gap", self.gap_var, 0, 9999, "frames")
        self._spin(spins, "Pick", self.selects_var, 1, 50, "results")

        actions = tk.Frame(panel, bg=THEME.panel)
        actions.pack(fill="x", pady=(18, 0))
        self.run_btn = tk.Button(
            actions,
            text="Run Extraction",
            command=self._run,
            bg=THEME.accent,
            fg=THEME.bg,
            activebackground="#ff9f61",
            activeforeground=THEME.bg,
            relief="flat",
            font=("Avenir Next", 11, "bold"),
            padx=16,
            pady=12,
            cursor="hand2",
        )
        self.run_btn.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.cancel_btn = tk.Button(
            actions,
            text="Cancel",
            command=self._cancel,
            bg=THEME.panel_alt,
            fg=THEME.text_muted,
            activebackground=THEME.border,
            activeforeground=THEME.text,
            relief="flat",
            font=("Avenir Next", 11),
            padx=16,
            pady=12,
            cursor="hand2",
            state="disabled",
        )
        self.cancel_btn.pack(side="left")

    def _build_status_panel(self, parent):
        status = self._card(parent, "Session", "A quick read on mode, validation, and processing state.")
        status.grid(row=0, column=0, sticky="ew")

        tk.Label(status, textvariable=self.summary_var, bg=THEME.panel, fg=THEME.text, font=("Avenir Next", 12)).pack(anchor="w")
        tk.Label(status, textvariable=self.status_var, bg=THEME.panel, fg=THEME.text_muted, font=("Avenir Next", 10)).pack(anchor="w", pady=(8, 10))

        self.progress = ttk.Progressbar(
            status,
            mode="determinate",
            style="Candid.Horizontal.TProgressbar",
        )
        self.progress.pack(fill="x")

    def _build_results_panel(self, parent):
        self.results = self._card(parent, "Results", "Click any frame to inspect it at full size.", inner_pad=False)
        self.results.grid(row=1, column=0, sticky="nsew", pady=(18, 0))

        top = tk.Frame(self.results, bg=THEME.panel)
        top.pack(fill="x", padx=18, pady=(18, 12))
        top.columnconfigure(0, weight=1)

        tk.Label(top, text="No frames yet.", bg=THEME.panel, fg=THEME.text_muted, font=("Avenir Next", 11)).grid(row=0, column=0, sticky="w")
        self.save_all_btn = tk.Button(
            top,
            text="Save All",
            command=self._download_all,
            bg=THEME.panel_alt,
            fg=THEME.text_muted,
            activebackground=THEME.border,
            activeforeground=THEME.text,
            relief="flat",
            font=("Avenir Next", 10),
            cursor="hand2",
            state="disabled",
            padx=12,
            pady=8,
        )
        self.save_all_btn.grid(row=0, column=1, sticky="e")

        wrapper = tk.Frame(self.results, bg=THEME.panel)
        wrapper.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        wrapper.rowconfigure(0, weight=1)
        wrapper.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(wrapper, bg=THEME.panel, highlightthickness=0)
        scrollbar = ttk.Scrollbar(wrapper, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.grid_frame = tk.Frame(self.canvas, bg=THEME.panel)
        self.grid_window = self.canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        self.grid_frame.bind("<Configure>", lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda event: self.canvas.itemconfigure(self.grid_window, width=event.width))

        self.bind_all("<MouseWheel>", self._on_scroll)

    def _card(self, parent, title: str, subtitle: str, inner_pad: bool = True):
        outer = tk.Frame(parent, bg=THEME.panel, highlightthickness=1, highlightbackground=THEME.border)
        header = tk.Frame(outer, bg=THEME.panel)
        header.pack(fill="x", padx=18, pady=(16, 4))
        tk.Label(header, text=title, bg=THEME.panel, fg=THEME.text, font=("Avenir Next", 14, "bold")).pack(anchor="w")
        tk.Label(header, text=subtitle, bg=THEME.panel, fg=THEME.text_soft, font=("Avenir Next", 10)).pack(anchor="w", pady=(4, 0))
        if inner_pad:
            tk.Frame(outer, bg=THEME.panel).pack(fill="x", padx=18, pady=(0, 14))
        return outer

    def _field(self, parent, label, var, browse=None, secret=False, placeholder=""):
        wrap = tk.Frame(parent, bg=THEME.panel)
        wrap.pack(fill="x", padx=18, pady=(0, 10))
        tk.Label(wrap, text=label, bg=THEME.panel, fg=THEME.text_muted, font=("Avenir Next", 10, "bold")).pack(anchor="w", pady=(0, 5))

        row = tk.Frame(wrap, bg=THEME.panel)
        row.pack(fill="x")

        entry = tk.Entry(
            row,
            textvariable=var,
            bg=THEME.panel_alt,
            fg=THEME.text,
            insertbackground=THEME.text,
            relief="flat",
            highlightthickness=1,
            highlightbackground=THEME.border,
            highlightcolor=THEME.accent,
            font=("Avenir Next", 11),
            show="*" if secret else "",
        )
        entry.pack(side="left", fill="x", expand=True, ipady=9)
        if placeholder and not var.get():
            entry.insert(0, placeholder)
            entry.config(fg=THEME.text_soft)

            def clear_placeholder(_event):
                if entry.get() == placeholder and entry.cget("fg") == THEME.text_soft:
                    entry.delete(0, "end")
                    entry.config(fg=THEME.text, show="*" if secret else "")

            def restore_placeholder(_event):
                if not entry.get():
                    entry.insert(0, placeholder)
                    entry.config(fg=THEME.text_soft, show="")

            entry.bind("<FocusIn>", clear_placeholder)
            entry.bind("<FocusOut>", restore_placeholder)

        if browse:
            tk.Button(
                row,
                text="Browse",
                command=browse,
                bg=THEME.panel_alt,
                fg=THEME.text,
                activebackground=THEME.border,
                activeforeground=THEME.text,
                relief="flat",
                font=("Avenir Next", 10, "bold"),
                padx=12,
                pady=9,
                cursor="hand2",
            ).pack(side="left", padx=(8, 0))

    def _meta_block(self, parent):
        box = tk.Frame(parent, bg=THEME.panel_alt, highlightthickness=1, highlightbackground=THEME.border)
        box.pack(fill="x", padx=18, pady=(2, 14))
        tk.Label(box, text="Detected video metadata", bg=THEME.panel_alt, fg=THEME.text_muted, font=("Avenir Next", 9, "bold")).pack(anchor="w", padx=12, pady=(10, 4))
        tk.Label(box, textvariable=self.meta_var, bg=THEME.panel_alt, fg=THEME.text, font=("Avenir Next", 10), wraplength=300, justify="left").pack(anchor="w", padx=12, pady=(0, 10))

    def _mode_card(self, parent, title, body, footnote):
        card = tk.Frame(parent, bg=THEME.panel_alt, highlightthickness=1, highlightbackground=THEME.border)
        tk.Label(card, text=title, bg=THEME.panel_alt, fg=THEME.text, font=("Avenir Next", 11, "bold")).pack(anchor="w", padx=12, pady=(12, 3))
        tk.Label(card, text=body, bg=THEME.panel_alt, fg=THEME.text_muted, font=("Avenir Next", 10), wraplength=280, justify="left").pack(anchor="w", padx=12)
        tk.Label(card, text=footnote, bg=THEME.panel_alt, fg=THEME.accent_soft, font=("Avenir Next", 9), wraplength=280, justify="left").pack(anchor="w", padx=12, pady=(8, 12))
        return card

    def _spin(self, parent, label, var, low, high, suffix):
        wrap = tk.Frame(parent, bg=THEME.panel)
        wrap.pack(fill="x", pady=(0, 8))
        tk.Label(wrap, text=f"{label} ({suffix})", bg=THEME.panel, fg=THEME.text_muted, font=("Avenir Next", 10, "bold")).pack(anchor="w")
        tk.Spinbox(
            wrap,
            from_=low,
            to=high,
            textvariable=var,
            bg=THEME.panel_alt,
            fg=THEME.text,
            insertbackground=THEME.text,
            relief="flat",
            highlightthickness=1,
            highlightbackground=THEME.border,
            highlightcolor=THEME.accent,
            buttonbackground=THEME.panel_alt,
            font=("Avenir Next", 11),
            width=8,
        ).pack(anchor="w", pady=(4, 0), ipady=6)

    def _on_scroll(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _browse_video(self):
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"), ("All files", "*.*")],
        )
        if path:
            self.video_var.set(path)
            self._load_metadata(path)

    def _load_metadata(self, path: str):
        try:
            meta = get_video_metadata(path)
        except Exception:
            self.meta_var.set("Unable to read this file.")
            return
        mins, secs = divmod(int(meta.duration_s), 60)
        self.meta_var.set(f"{meta.width}x{meta.height}  |  {meta.fps:.2f} fps  |  {mins}m {secs:02d}s  |  {meta.frames:,} frames")
        self.summary_var.set("Ready for smart curation. Add four prompts plus an OpenAI key for question-strip mode.")

    def _build_options(self) -> ExtractionOptions:
        questions = tuple(var.get().strip() for var in self.q_vars if var.get().strip())
        return ExtractionOptions(
            sample_every=self.sample_var.get(),
            top_n=self.top_var.get(),
            min_gap=self.gap_var.get(),
            n_select=max(1, self.selects_var.get()),
            gemini_key=self.gemini_key_var.get().strip(),
            openai_key=self.openai_key_var.get().strip(),
            questions=questions,
        )

    def _validate(self, video_path: str, options: ExtractionOptions):
        if not video_path:
            raise ValueError("Choose a video file first.")
        if not os.path.isfile(video_path):
            raise ValueError(f"Video not found:\n{video_path}")
        if not options.gemini_key:
            raise ValueError("Add your Gemini API key before running.")
        raw_questions = [var.get().strip() for var in self.q_vars]
        filled_questions = [question for question in raw_questions if question]
        if (filled_questions or options.openai_key) and not options.use_question_mode:
            raise ValueError(
                "Question-strip mode needs all four question prompts plus an OpenAI API key. "
                "Clear those fields to use smart curation instead."
            )

    def _run(self):
        video_path = self.video_var.get().strip()
        options = self._build_options()
        try:
            self._validate(video_path, options)
        except ValueError as exc:
            messagebox.showerror("Cannot start extraction", str(exc))
            return

        self._cancel_event.clear()
        self._frame_data.clear()
        self._clear_grid()
        self.progress.configure(value=0)
        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.save_all_btn.configure(state="disabled")
        self._last_video_path = video_path

        if options.use_question_mode:
            self.summary_var.set("Question-strip mode active: one final frame per answer.")
        else:
            self.summary_var.set("Smart curation mode active: scanning the whole video for standout stills.")

        threading.Thread(target=self._pipeline_worker, args=(video_path, options), daemon=True).start()

    def _pipeline_worker(self, video_path: str, options: ExtractionOptions):
        try:
            results = run_pipeline(video_path, options, self._update_progress, self._cancel_event)
            if self._cancel_event.is_set():
                self._update_progress("Cancelled.", 0)
                return
            self.after(0, lambda: self._show_results(results))
        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("Processing failed", str(exc)))
            self._update_progress(f"Error: {exc}", 0)
        finally:
            self.after(0, lambda: self.run_btn.configure(state="normal"))
            self.after(0, lambda: self.cancel_btn.configure(state="disabled"))

    def _cancel(self):
        self._cancel_event.set()
        self.cancel_btn.configure(state="disabled")
        self.status_var.set("Cancelling current run...")

    def _update_progress(self, message, value):
        self.after(0, lambda: self.status_var.set(message or self.status_var.get()))
        if value is not None:
            self.after(0, lambda: self.progress.configure(value=value))

    def _clear_grid(self):
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        self._tk_images.clear()

    def _show_results(self, frames: list[SelectedFrame]):
        self._frame_data = frames
        self._clear_grid()
        self.save_all_btn.configure(state="normal" if frames else "disabled")
        self.summary_var.set(f"Finished. {len(frames)} frame(s) ready to review and save.")
        if not frames:
            self.status_var.set("No frames were selected.")
            return

        for col in range(DEFAULTS.grid_cols):
            self.grid_frame.columnconfigure(col, weight=1)

        for idx, frame in enumerate(frames):
            row, col = divmod(idx, DEFAULTS.grid_cols)
            card = tk.Frame(self.grid_frame, bg=THEME.panel_alt, highlightthickness=1, highlightbackground=THEME.border)
            card.grid(row=row, column=col, sticky="nsew", padx=8, pady=8)

            rgb = cv2.cvtColor(frame.frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image.thumbnail((500, 320), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(image)
            self._tk_images.append(tk_img)

            image_label = tk.Label(card, image=tk_img, bg=THEME.panel_alt, cursor="hand2")
            image_label.pack(fill="both", expand=True, padx=10, pady=(10, 8))
            image_label.bind("<Button-1>", lambda _event, data=frame, i=idx: self._zoom_frame(data, i))

            text = frame.question_label or f"Selection {idx + 1}"
            subtitle = f"Frame {frame.frame_index:,}  |  Sharpness {frame.score:,.0f}"
            tk.Label(card, text=text, bg=THEME.panel_alt, fg=THEME.text, font=("Avenir Next", 11, "bold"), wraplength=420, justify="left").pack(anchor="w", padx=12)
            tk.Label(card, text=subtitle, bg=THEME.panel_alt, fg=THEME.text_muted, font=("Avenir Next", 10)).pack(anchor="w", padx=12, pady=(4, 10))

            tk.Button(
                card,
                text="Save Frame",
                command=lambda i=idx: self._download_single(i),
                bg=THEME.accent,
                fg=THEME.bg,
                activebackground="#ff9f61",
                activeforeground=THEME.bg,
                relief="flat",
                font=("Avenir Next", 10, "bold"),
                padx=12,
                pady=8,
                cursor="hand2",
            ).pack(anchor="w", padx=12, pady=(0, 12))

    def _zoom_frame(self, frame: SelectedFrame, idx: int):
        ZoomWindow(self, frame.frame_bgr, title=f"Frame {idx + 1} | source frame {frame.frame_index}")

    def _save_path(self, initial_file: str):
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
        frame = self._frame_data[idx]
        path = self._save_path(f"frame_{idx + 1}_{frame.frame_index}.jpg")
        if not path:
            return
        self._write_frame(path, frame.frame_bgr)
        self.status_var.set(f"Saved {os.path.basename(path)}")

    def _download_all(self):
        if not self._frame_data or not self._last_video_path:
            return
        video_root = os.path.splitext(os.path.basename(self._last_video_path))[0]
        folder = os.path.join(os.path.dirname(os.path.abspath(self._last_video_path)), video_root)
        os.makedirs(folder, exist_ok=True)
        for idx, frame in enumerate(self._frame_data, start=1):
            self._write_frame(os.path.join(folder, f"frame_{idx}_{frame.frame_index}.jpg"), frame.frame_bgr)
        self._last_save_dir = folder
        self.status_var.set(f"Saved {len(self._frame_data)} frames to {folder}")
        messagebox.showinfo("Save All", f"Saved {len(self._frame_data)} frames to:\n{folder}")


def run_cli(args: argparse.Namespace) -> int:
    options = ExtractionOptions(
        sample_every=args.sample_every,
        top_n=args.top_n,
        min_gap=args.min_gap,
        n_select=args.count,
        gemini_key=args.gemini_key,
        openai_key=args.openai_key or "",
        questions=tuple(args.question or []),
    )
    results = run_pipeline(args.video, options)
    out_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(args.video)), os.path.splitext(os.path.basename(args.video))[0])
    os.makedirs(out_dir, exist_ok=True)
    for idx, frame in enumerate(results, start=1):
        path = os.path.join(out_dir, f"frame_{idx}_{frame.frame_index}.jpg")
        if path.lower().endswith(".jpg"):
            cv2.imwrite(path, frame.frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(path, frame.frame_bgr)
        print(path)
    return 0
