import argparse

from .config import PipelineDefaults
from .platform_checks import ensure_supported_tk
from .ui import App, run_cli


DEFAULTS = PipelineDefaults()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract curated still frames from a video.")
    parser.add_argument("--cli", action="store_true", help="Run without the desktop UI.")
    parser.add_argument("--video", help="Path to the source video.")
    parser.add_argument("--gemini-key", default="", help="Gemini API key.")
    parser.add_argument("--openai-key", default="", help="OpenAI API key for question mode.")
    parser.add_argument("--question", action="append", help="Interview question prompt. Provide this four times for question mode.")
    parser.add_argument("--output-dir", default="", help="Directory to save exported frames in CLI mode.")
    parser.add_argument("--sample-every", type=int, default=DEFAULTS.sample_every_n)
    parser.add_argument("--top-n", type=int, default=DEFAULTS.top_n_sharp)
    parser.add_argument("--min-gap", type=int, default=DEFAULTS.temporal_gap)
    parser.add_argument("--count", type=int, default=DEFAULTS.gemini_selects)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.cli:
        if not args.video or not args.gemini_key:
            parser.error("--cli requires --video and --gemini-key")
        return run_cli(args)

    ensure_supported_tk()
    app = App()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

