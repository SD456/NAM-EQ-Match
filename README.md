# NAM EQ Matcher 🎸

Clone a guitar tone from a finished recording — no DI from the original player needed.

Combines **NAM** (nonlinear amp character: gain structure, saturation, compression, sag)
with a **Tone Match**-style matched impulse response (linear EQ/cab correction).

## How it works

You provide three things:

1. **Target** — a recording with the tone you want: an isolated guitar track, or a full mix
   (built-in Demucs demixing extracts the guitar stem for you)
2. **DI** — your own clean guitar take (any riff, ideally similar playing style/register)
3. **NAM library** — a folder of `.nam` captures, or use the built-in **TONE3000 search**
   to browse the catalog (metadata only) and download just a shortlist

The pipeline:

1. **Dynamic/saturation match (NAM half).** Your DI is reamped through every capture in
   the library across a grid of input gains. Each result is compared to the target using an
   *EQ-invariant fingerprint*: crest factor, dynamic range compression, spectral flatness of
   the fizz region, spectral flux, and MFCC texture statistics. These capture what an EQ
   cannot fix — clipping behavior, compression feel, harmonic density. A two-stage search
   (coarse over all captures, fine over the top 5) finds the best capture + drive setting.
2. **Frequency match (Tone Match half).** The winning rig's output is compared to the
   target's 1/6-octave smoothed long-term spectrum, and the gap is closed three ways —
   pick by ear:
   - **A: Match IR** (`_match_ir.wav`) — the exact spectral correction (±18 dB). Closest
     match, but drastic corrections can sound artificial.
   - **B: Plugin EQ suggestion** — fitted Bass/Mid/Treble knob values for the NAM
     plugin's built-in tone stack (exact same filters: 150 Hz shelf / 425 Hz peak /
     1.8 kHz shelf). Gentler and hardware-plausible; no IR needed.
   - **C: Hybrid (recommended)** — the EQ settings plus a *gentle* residual IR
     (`_gentle_ir.wav`, capped at ±9 dB, broad 1/3-octave smoothing). Natural EQ moves
     do the heavy lifting; the IR only polishes what's left.
3. **Outputs.** Best model + input gain, the match IR, a rendered preview of *your* DI
   through the full matched chain, a comparison spectrum plot, and a JSON report.
   Set *Top rigs to render* (GUI) or `--render-top N` (CLI) to get the top-N captures
   instead of just the winner. Every rendered rig gets self-explaining, rig-prefixed
   files (`rank01_JCM800_IN+2.0_...`): a copy of the `.nam`, a human-readable
   `_settings.txt` with Options A/B/C, both IRs, and all three renders — so files stay
   meaningful even when moved out of the results folder.

## Quick start (no command line needed)

Install Python 3.10+ from [python.org](https://www.python.org/downloads/) (on Windows,
tick *Add Python to PATH*), then double-click the launcher for your system:

- **Windows**: `Start-NAM-EQ-Matcher-Windows.bat`
- **macOS**: `start-nam-eq-matcher-macos.command` (first time: right-click → Open;
  if blocked, run `chmod +x start-nam-eq-matcher-macos.command` once in Terminal)
- **Linux**: `bash start-nam-eq-matcher-linux.sh`

The first run creates a private environment and installs everything (several minutes);
after that it starts in seconds and opens the app in your browser automatically.

## Manual install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Use

**GUI:**

```bash
python app.py
# open http://127.0.0.1:7860
```

**CLI:**

```bash
python match.py --target song_stem.wav --di my_di.wav --models ./nam_models --out ./results

# render the top 5 rigs, each with its own match IR:
python match.py --target song_stem.wav --di my_di.wav --models ./nam_models --render-top 5

# full mix? extract the guitar stem first with Demucs:
python match.py --target full_song.mp3 --demix --di my_di.wav --models ./nam_models --out ./results
```

**TONE3000 search (no bulk downloads):**

Create a free publishable API key at tone3000.com → Settings → API Keys (`t3k_pub_...`).
Searching returns metadata only; you download just the tones you pick (a `.nam` is usually
well under 1 MB). Matching itself always needs the files — the matcher must reamp your DI
through each capture. First connect opens your browser for TONE3000 login (OAuth); tokens
are cached in `~/.tonematch/`. Rate limit: 100 requests/min.

In the GUI: open the *Search TONE3000* accordion → connect → search → enter row numbers →
Download selected (goes to `./t3k_cache`, and the models-folder field is filled in for you).

If you know what amps you're after, list them in the *Amps you're looking for* field
(comma-separated, up to 6). NAM EQ Matcher runs one catalog query per amp, merges the results,
and ranks tones whose **make metadata** matches first (then title, tags, description) — the
*Matched on* column shows why each result surfaced.

CLI:

```bash
python -m tonematch.tone3000 connect --key t3k_pub_xxx
python -m tonematch.tone3000 search "crunch" --amps "Marshall,JCM800,Friedman" --gear amp
python -m tonematch.tone3000 download 12345 67890 --out ./t3k_cache
python match.py --target stem.wav --di my_di.wav --models ./t3k_cache
```

**Test (no torch needed):**

```bash
python -m tests.test_synthetic        # end-to-end pipeline
python -m tests.test_drive_matching   # driven targets must not match clean captures
```

## GUI workflow (tabs)

- **Match** — set up target/DI/models and *queue* a run. You can queue several with
  different settings; they execute one at a time in the background.
- **Library** — scan a folder of `.nam` files (recursive, cached), filter by name, make,
  tone type (`clean`/`crunch`/`hi_gain`/...) or gear type, and save the filtered set as a
  named library selectable in the Match tab.
- **Jobs** — live queue view (auto-refreshes every 2 s); cancel queued jobs.
- **Results** — pick any completed job, A/B/C/D the renders, and export a ZIP of exactly
  the rigs and file types you want.

Last-used folders and settings are remembered between sessions (`~/.tonematch/`).
The job queue, library curation, ZIP export and session persistence were suggested by a
community contributor (PR #2) — thanks! — and re-implemented for the current codebase.

## Using the result

In the NAM plugin: load the winning `.nam` file, set **Input** to the reported gain (dB),
load `match_ir.wav` in the **IR slot** (disable any other cab), adjust Output to taste.

## Tips

- The better your target isolation, the better the match. For full mixes use `--demix`
  (or the GUI checkbox): Demucs `htdemucs_6s` extracts a dedicated **guitar** stem. If the
  guitar sounds thin/incomplete in that stem, try `guitar+other`. First run downloads the
  model; CPU separation takes roughly the length of the song. Expect residual bleed to
  slightly skew the EQ match — trust your ears over the plot.
- Play a DI similar in register and intensity to the target part — the matcher compares
  statistics, not aligned samples, but similar material makes them comparable.
- Renders use the loudest 30 s of your DI by default (*Render length* in Advanced /
  `--preview-s`). The IRs and EQ fits are statistical and converge within ~20–30 s, so
  this is a large speedup on long DIs at no accuracy cost; set 0 to render the full DI.
- More diverse `.nam` libraries → better odds one capture nails the clipping character.
- `report.json` includes a **`drive_character`** section: a 0–1 "how driven does this
  sound" index for the target and for the chosen rig. If those differ a lot, your library
  has no capture with the right gain structure — add captures of that amp type rather
  than trusting the match.
- Search is compute-heavy: ~10 reamps per capture. Use "Max captures" or a curated subfolder
  for quick passes.
- **GPU acceleration**: NAM inference and Demucs both use CUDA automatically when your
  PyTorch build supports it (device "auto"; force with the device dropdown / `--device cuda`).
  The default `pip install torch` on Windows is CPU-only — for a ~10× speedup, install a
  CUDA build instead, e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu126`
  (pick the right CUDA version for your GPU at pytorch.org/get-started). Verify with
  `python -m tonematch.doctor` — it should print `torch device: cuda`.

## Project layout

The Python package keeps the internal name `tonematch` (renaming it would break
imports and your venv); the product name is **NAM EQ Matcher**.

```
tonematch/
  audio.py        I/O, resampling, envelopes, segment selection
  features.py     tone fingerprint (dynamics, saturation, texture) + LTAS
  match_eq.py     matched-IR design (1/6-oct smoothing, min-phase FIR)
  nam_backend.py  .nam loading (neural-amp-modeler) + MockAmp for tests
  search.py       two-stage capture ranking + input-gain search
  tone_stack.py   NAM plugin tone stack (B/M/T) emulation + knob fitting
  jobs.py         background job queue for match runs
  library.py      .nam scanning/filtering + named libraries
  stems.py        Demucs guitar-stem extraction (optional dependency)
  tone3000.py     TONE3000 API client (OAuth PKCE, search, selective download)
  pipeline.py     end-to-end orchestration + reports
app.py            Gradio GUI
match.py          CLI
tests/            synthetic end-to-end test
```
