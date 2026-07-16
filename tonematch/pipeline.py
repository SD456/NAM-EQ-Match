"""End-to-end tone match pipeline.

Given:
  * target: an isolated guitar recording (the tone you want)
  * di:     your clean DI (or any clean guitar performance)
  * a library of .nam captures

Per rendered rig, three ways to close the EQ gap (from most surgical to most
natural-sounding):

  A. Match IR      - exact spectral match; can sound "artificial" if the
                     correction is drastic.
  B. Plugin EQ     - suggested Bass/Mid/Treble knob values for the NAM
                     plugin's built-in tone stack. Gentler, hardware-plausible.
  C. Hybrid        - plugin EQ first, then a *gentle* residual IR (capped at
                     +/-9 dB, broader 1/3-octave smoothing). Usually the best
                     compromise.

Outputs per rig: .nam copy, match IR, gentle IR, three renders, a settings
text file, plus a JSON report and comparison plot.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field

import numpy as np

from .audio import active_segment, db_to_lin, lin_to_db, load_audio, match_rms, rms, save_audio
from .features import extract_fingerprint
from .match_eq import apply_fir, design_match_ir
from .search import CandidateResult, rank_captures
from .tone_stack import apply_tone_stack, fit_tone_stack

DEFAULT_SR = 48000


@dataclass
class MatchOutput:
    ranked: list
    best: CandidateResult
    ir_path: str
    render_path: str
    target_ref_path: str
    report_path: str
    plot_path: str | None
    report: dict
    renders: list = field(default_factory=list)  # one dict per rendered rig


def _finalize(audio: np.ndarray, target_rms: float) -> tuple[np.ndarray, float]:
    """RMS-match to target level with peak safety. Returns (audio, gain_lin)."""
    out, gain = match_rms(audio, target_rms)
    peak = np.max(np.abs(out)) + 1e-9
    if peak > 0.99:
        out *= 0.99 / peak
        gain *= 0.99 / peak
    return out, gain


def run_match(
    target_path: str,
    di_path: str,
    captures: list,
    out_dir: str,
    sr: int = DEFAULT_SR,
    gain_range_db: tuple[float, float] = (-12.0, 12.0),
    refine_top: int = 5,
    render_top: int = 1,
    ir_taps: int = 2048,
    preview_s: float = 30.0,
    progress_cb=None,
) -> MatchOutput:
    os.makedirs(out_dir, exist_ok=True)

    def report_progress(frac, msg):
        if progress_cb:
            progress_cb(frac, msg)

    report_progress(0.0, "loading audio")
    target, _ = load_audio(target_path, sr)
    di, _ = load_audio(di_path, sr)
    # normalize DI to a sane nominal level (~ -18 dBFS RMS) so gain search is meaningful
    di, _ = match_rms(di, 10 ** (-18 / 20))
    target_peak_norm = target / (np.max(np.abs(target)) + 1e-9)
    ref_rms = rms(target_peak_norm) * 0.5

    # --- 1) rank captures & find input gain (the NAM / dynamics half) -------
    ranked = rank_captures(
        captures,
        di,
        target,
        sr,
        gain_range_db=gain_range_db,
        refine_top=refine_top,
        progress_cb=lambda f, m: report_progress(0.05 + 0.72 * f, m),
    )
    if not ranked:
        raise ValueError("No captures could be evaluated - check your .nam files/folder.")
    best = ranked[0]
    target_fp = extract_fingerprint(target, sr)

    # --- 2) render the top-K rigs: match IR, plugin EQ, and hybrid ----------
    n_render = max(1, min(int(render_top), len(ranked)))
    renders = []
    plot_data = None
    for i, r in enumerate(ranked[:n_render]):
        report_progress(0.8 + 0.15 * i / n_render, f"rendering rig {i + 1}/{n_render}: {r.name}")
        safe = re.sub(r"[^A-Za-z0-9]+", "-", r.name).strip("-")[:40] or f"rig{i + 1}"
        prefix = f"rank{i + 1:02d}_{safe}_IN{r.gain_db:+.1f}"

        def rig_path(suffix):
            return os.path.join(out_dir, f"{prefix}{suffix}")

        # Reamp only the most active `preview_s` seconds of the DI (0 = full DI).
        # IRs and EQ fits use long-term spectral statistics, which converge well
        # within ~20-30 s - this slashes per-rig render time on long DIs.
        di_render = di if not preview_s else active_segment(di, sr, float(preview_s))
        amped = r.capture.process(di_render * db_to_lin(r.gain_db))
        amped_fp = extract_fingerprint(amped, sr)

        # A) full match IR on the raw amp output
        ir, grid, corr_db = design_match_ir(amped, target, sr, n_taps=ir_taps)
        ir_rendered, out_gain = _finalize(apply_fir(amped, ir), ref_rms)

        # B) NAM plugin tone stack (Bass/Mid/Treble) suggestion + render
        fit = fit_tone_stack(amped_fp.ltas_db, target_fp.ltas_db, amped_fp.ltas_grid, sr)
        eq_audio = apply_tone_stack(amped, sr, fit.bass, fit.middle, fit.treble)
        eq_rendered, eq_out_gain = _finalize(eq_audio, ref_rms)

        # C) hybrid: tone stack + gentle residual IR (capped, broadly smoothed)
        gentle_ir, _, gentle_corr = design_match_ir(
            eq_audio, target, sr, n_taps=ir_taps, max_gain_db=9.0, octave_fraction=3.0
        )
        hybrid_rendered, hy_out_gain = _finalize(apply_fir(eq_audio, gentle_ir), ref_rms)

        # --- write rig artifacts (rig-prefixed, self-explaining names) -------
        eq_tag = f"_EQ_B{fit.bass:g}_M{fit.middle:g}_T{fit.treble:g}"
        ir_path_ = rig_path("_match_ir.wav")
        render_path_ = rig_path("_matched_render.wav")
        eq_render_path = rig_path(f"{eq_tag}_render.wav")
        gentle_ir_path = rig_path("_gentle_ir.wav")
        hybrid_path = rig_path(f"{eq_tag}_plus_gentle_ir_render.wav")
        save_audio(ir_path_, ir, sr)
        save_audio(render_path_, ir_rendered, sr)
        save_audio(eq_render_path, eq_rendered, sr)
        save_audio(gentle_ir_path, gentle_ir, sr)
        save_audio(hybrid_path, hybrid_rendered, sr)

        nam_copy = None
        src = getattr(r.capture, "path", None)
        if src and os.path.isfile(src):
            nam_copy = rig_path(".nam")
            shutil.copyfile(src, nam_copy)

        entry = {
            "rank": i + 1,
            "name": r.name,
            "file": src,
            "nam_copy": os.path.abspath(nam_copy) if nam_copy else None,
            "input_gain_db": round(r.gain_db, 2),
            "output_gain_db": round(lin_to_db(out_gain), 2),
            "score": round(r.score, 4),
            "ir": os.path.abspath(ir_path_),
            "render": os.path.abspath(render_path_),
            "tone_stack": {
                "bass": fit.bass,
                "middle": fit.middle,
                "treble": fit.treble,
                "bass_gain_db": fit.bass_gain_db,
                "middle_gain_db": fit.middle_gain_db,
                "treble_gain_db": fit.treble_gain_db,
                "ltas_mse_before": fit.mse_before,
                "ltas_mse_after": fit.mse_after,
                "render": os.path.abspath(eq_render_path),
            },
            "hybrid": {
                "gentle_ir": os.path.abspath(gentle_ir_path),
                "render": os.path.abspath(hybrid_path),
            },
        }
        settings_path = rig_path("_settings.txt")
        _write_settings_txt(settings_path, entry, fit)
        entry["settings_txt"] = os.path.abspath(settings_path)
        renders.append(entry)

        if i == 0:
            plot_data = dict(
                rendered=ir_rendered, grid=grid, corr_db=corr_db,
                amped=amped, eq_rendered=eq_rendered, fit=fit,
            )

    # --- 3) shared artifacts --------------------------------------------------
    report_progress(0.96, "writing outputs")
    target_ref_path = os.path.join(out_dir, "target_reference.wav")
    save_audio(target_ref_path, target_peak_norm * 0.5, sr)
    ir_path = renders[0]["ir"]
    render_path = renders[0]["render"]

    plot_path = None
    try:
        plot_path = _plot_match(os.path.join(out_dir, "match_plot.png"), target, sr, **plot_data)
    except Exception as e:  # matplotlib optional
        print(f"[warn] plot skipped: {e}")

    ts0 = renders[0]["tone_stack"]
    report = {
        "best_model": {
            "name": best.name,
            "file": getattr(best.capture, "path", None),
            "input_gain_db": round(best.gain_db, 2),
            "output_gain_db": renders[0]["output_gain_db"],
            "score": round(best.score, 4),
            "nonlinear_distance": round(best.nl_distance, 4),
            "eq_penalty": round(best.eq_penalty, 4),
            "breakdown": {k: round(v, 4) for k, v in best.breakdown.items()},
        },
        "ranking": [
            {
                "name": r.name,
                "file": getattr(r.capture, "path", None),
                "input_gain_db": round(r.gain_db, 2),
                "score": round(r.score, 4),
            }
            for r in ranked
        ],
        "renders": renders,
        "match_ir": os.path.abspath(ir_path),
        "how_to_use": (
            "In the NAM plugin: load the model file, set Input gain to "
            f"{best.gain_db:+.1f} dB, load match_ir.wav in the IR slot "
            "(disable any other cab), then adjust Output to taste. "
            "Alternative (more natural, less surgical): skip the match IR, enable the "
            f"plugin's EQ and set Bass {ts0['bass']:g}, Middle {ts0['middle']:g}, "
            f"Treble {ts0['treble']:g}. Or combine: EQ settings + gentle_ir.wav in the IR slot."
        ),
        "sample_rate": sr,
    }
    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    report_progress(1.0, "done")
    return MatchOutput(
        ranked=ranked,
        best=best,
        ir_path=os.path.abspath(ir_path),
        render_path=os.path.abspath(render_path),
        target_ref_path=os.path.abspath(target_ref_path),
        report_path=os.path.abspath(report_path),
        plot_path=os.path.abspath(plot_path) if plot_path else None,
        report=report,
        renders=renders,
    )


def _write_settings_txt(path: str, entry: dict, fit) -> None:
    ts = entry["tone_stack"]
    lines = [
        "NAM EQ Matcher - rig settings",
        "",
        f"Rank: #{entry['rank']}",
        f"NAM model: {entry['name']}",
        f"Original file: {entry['file']}",
        "",
        f"Input gain:  {entry['input_gain_db']:+.1f} dB",
        f"Output gain: {entry['output_gain_db']:+.1f} dB (approximate - set to taste)",
        "",
        "Option A - Match IR (closest spectral match):",
        f"  Load the .nam, set Input to {entry['input_gain_db']:+.1f} dB,",
        "  load the _match_ir.wav in the IR slot, disable any other cab.",
        "",
        "Option B - Plugin EQ only (most natural):",
        f"  Load the .nam, set Input to {entry['input_gain_db']:+.1f} dB, disable the IR,",
        "  enable the plugin EQ and set:",
        f"    Bass   {ts['bass']:g}   ({ts['bass_gain_db']:+.2f} dB @ 150 Hz shelf)",
        f"    Middle {ts['middle']:g}   ({ts['middle_gain_db']:+.2f} dB @ 425 Hz peak)",
        f"    Treble {ts['treble']:g}   ({ts['treble_gain_db']:+.2f} dB @ 1.8 kHz shelf)",
        "",
        "Option C - Hybrid (recommended):",
        "  Option B settings + load the _gentle_ir.wav in the IR slot.",
        "",
        "Files:",
        f"  NAM copy:         {os.path.basename(entry['nam_copy']) if entry['nam_copy'] else '(source was not a file)'}",
        f"  Match IR:         {os.path.basename(entry['ir'])}",
        f"  Gentle IR:        {os.path.basename(entry['hybrid']['gentle_ir'])}",
        f"  IR render:        {os.path.basename(entry['render'])}",
        f"  EQ-only render:   {os.path.basename(ts['render'])}",
        f"  Hybrid render:    {os.path.basename(entry['hybrid']['render'])}",
        "",
    ]
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))


def _plot_match(path, target, sr, rendered, grid, corr_db, amped=None, eq_rendered=None, fit=None):
    """Thread-safe spectrum plot.

    Uses the matplotlib object-oriented API directly (Figure / FigureCanvasAgg)
    instead of `pyplot`, which keeps a global current-figure stack that is not
    safe under concurrent calls. Safe to call from multiple threads in parallel.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from .tone_stack import band_response_db, stack_response_db

    fp_t = extract_fingerprint(target, sr)
    fp_r = extract_fingerprint(rendered, sr)

    n_rows = 3 if fit is not None else 2

    fig = Figure(figsize=(9, 3.2 * n_rows))
    FigureCanvasAgg(fig)
    axes = fig.subplots(n_rows, 1, sharex=True)

    ax1 = axes[0]
    ax1.semilogx(fp_t.ltas_grid, fp_t.ltas_db, label="target", lw=2)
    ax1.semilogx(fp_r.ltas_grid, fp_r.ltas_db, label="match IR render", lw=1.5, alpha=0.85)
    if amped is not None:
        fp_a = extract_fingerprint(amped, sr)
        ax1.semilogx(fp_a.ltas_grid, fp_a.ltas_db, "--", label="raw NAM output", lw=1.2, alpha=0.7)
    if eq_rendered is not None:
        fp_e = extract_fingerprint(eq_rendered, sr)
        ax1.semilogx(fp_e.ltas_grid, fp_e.ltas_db, "-.", label="plugin EQ render", lw=1.2, alpha=0.8)
    ax1.set_ylabel("LTAS (dB, median-norm.)")
    ax1.set_title("Tone match result")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)

    ax2 = axes[1]
    ax2.semilogx(grid, corr_db, color="tab:green", lw=1.5)
    ax2.set_ylabel("Match IR correction (dB)")
    ax2.grid(True, which="both", alpha=0.3)

    if fit is not None:
        ax3 = axes[2]
        for band, knob, color in (
            ("bass", fit.bass, "tab:brown"),
            ("middle", fit.middle, "tab:olive"),
            ("treble", fit.treble, "tab:cyan"),
        ):
            ax3.semilogx(grid, band_response_db(band, knob, grid, sr), color=color,
                         lw=1.0, alpha=0.8, label=f"{band} {knob:g}")
        ax3.semilogx(grid, stack_response_db(grid, sr, fit.bass, fit.middle, fit.treble),
                     color="tab:purple", lw=2, label="total EQ")
        ax3.axhline(0.0, color="gray", lw=0.7, alpha=0.6)
        ax3.set_ylabel("Plugin EQ response (dB)")
        ax3.grid(True, which="both", alpha=0.3)
        ax3.legend(fontsize=8)

    axes[-1].set_xlabel("Frequency (Hz)")
    axes[-1].set_xlim(30, sr / 2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path
