# Candid Frame Extractor

A desktop app for pulling polished still frames from videos using OpenCV for sharpness filtering and Gemini for final curation.

## What It Does

- Smart curation mode scans the whole video and returns the strongest frames.
- Question-strip mode maps 4 prompts to answer windows and picks 1 frame per question.
- Results can be previewed, zoomed, and saved individually or all at once.

## Setup

From the project folder:

```bash
python3 -m venv .venv-tk
.venv-tk/bin/pip install -r requirements.txt
```

Then launch:

```bash
./run.sh
```

## Desktop Usage

1. Click `Browse` and choose a video file.
2. Paste your `Gemini API key`.
3. Click `Run Extraction`.

For question-strip mode:

1. Paste your `OpenAI API key`.
2. Fill all 4 question fields.
3. Click `Run Extraction`.

After processing:

- Click a frame to zoom in.
- Click `Save Frame` to export one image.
- Click `Save All` to export the whole set.

## CLI Usage

Run standard curation:

```bash
.venv-tk/bin/python frame_extractor.py --cli \
  --video /path/to/video.mp4 \
  --gemini-key YOUR_GEMINI_KEY
```

Run question-strip mode:

```bash
.venv-tk/bin/python frame_extractor.py --cli \
  --video /path/to/video.mp4 \
  --gemini-key YOUR_GEMINI_KEY \
  --openai-key YOUR_OPENAI_KEY \
  --question "Question 1" \
  --question "Question 2" \
  --question "Question 3" \
  --question "Question 4"
```

Optional flags:

- `--output-dir /path/to/export/folder`
- `--sample-every 30`
- `--top-n 150`
- `--min-gap 90`
- `--count 10`

## Requirements

- Python with Tcl/Tk 8.6+ for the desktop UI on macOS
- `ffmpeg` available on your PATH for question-strip transcription
- Gemini API key
- OpenAI API key for question-strip mode

## Project Structure

- [frame_extractor.py](/Users/gurucharanlingamallu/Documents/candid/frame_extractor.py): thin launcher
- [candid_app/main.py](/Users/gurucharanlingamallu/Documents/candid/candid_app/main.py): app and CLI entrypoint
- [candid_app/ui.py](/Users/gurucharanlingamallu/Documents/candid/candid_app/ui.py): desktop interface
- [candid_app/pipeline.py](/Users/gurucharanlingamallu/Documents/candid/candid_app/pipeline.py): extraction and AI pipeline
- [candid_app/models.py](/Users/gurucharanlingamallu/Documents/candid/candid_app/models.py): typed data models
