"""NAM EQ Matcher GUI (Gradio).

Run:  python app.py   then open http://127.0.0.1:7860

Tabs:
  Match    - set up a tone match and queue it
  Library  - scan/filter your .nam collection, save named libraries
  Jobs     - watch queued/running matches (auto-refreshes)
  Results  - audition finished jobs, export a ZIP of selected files

Job queue, library curation, ZIP export and session persistence were suggested
by a community contributor (PR #2) and re-implemented for this codebase.
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile

import gradio as gr

from tonematch import library as libmod
from tonematch.jobs import JobManager
from tonematch.nam_backend import load_captures, resolve_device
from tonematch.pipeline import run_match

T3K_CACHE = os.path.abspath("t3k_cache")
SESSION_PATH = os.path.join(os.path.expanduser("~"), ".tonematch", "last_session.json")
SESSION_KEYS = ("models_dir", "gain_lo", "gain_hi", "refine_top", "render_top", "device", "preview_s", "library")

jm = JobManager()

DESCRIPTION = """
# 🎸 NAM EQ Matcher — clone a guitar tone from a recording

1. **Target**: a recording with the tone you want — an isolated guitar track, or a **full mix**
   (tick *Extract guitar stem* and Demucs will demix it first).
2. **DI**: your own clean (unamped) guitar take.
3. **NAM library**: a folder of `.nam` captures, a saved library (Library tab), or search
   **TONE3000** below and download only the shortlist you want to try.

Runs are queued — start one, keep tweaking, and watch progress in the **Jobs** tab.
Audition and download everything in the **Results** tab.
"""


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------


def _load_session() -> dict:
    try:
        with open(SESSION_PATH, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        return {k: data[k] for k in SESSION_KEYS if k in data}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_session(**values) -> None:
    data = {k: v for k, v in values.items() if k in SESSION_KEYS}
    os.makedirs(os.path.dirname(SESSION_PATH), exist_ok=True)
    tmp = SESSION_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)
    os.replace(tmp, SESSION_PATH)


_SESSION = _load_session()


# ---------------------------------------------------------------------------
# TONE3000 integration
# ---------------------------------------------------------------------------

_t3k_client = None


def _get_t3k():
    global _t3k_client
    if _t3k_client is None:
        from tonematch.tone3000 import T3KClient

        _t3k_client = T3KClient()
    return _t3k_client


def t3k_connect(key):
    from tonematch.tone3000 import T3KError

    client = _get_t3k()
    try:
        client.connect(key.strip() or None)
        return "✅ Connected to TONE3000."
    except T3KError as e:
        return f"❌ {e}"


def t3k_search(query, amps_text, gear, sort):
    from tonematch.tone3000 import T3KError

    client = _get_t3k()
    amps = [a for a in (amps_text or "").split(",") if a.strip()]
    try:
        tones = client.search_tones_refined(query or "", amps=amps, gear=gear, sort=sort, page_size=30)
    except T3KError as e:
        raise gr.Error(str(e))
    rows = [
        [i + 1, t["id"], t["title"], ", ".join(m["name"] for m in t.get("makes", [])) or "-",
         t.get("_matched_on", "-"), t["gear"], t["models_count"], t["downloads_count"]]
        for i, t in enumerate(tones)
    ]
    hint = f" (refined by: {', '.join(a.strip() for a in amps)})" if amps else ""
    return rows, tones, f"{len(tones)} NAM tones found{hint}. Pick rows below to download."


def t3k_download(tones, rows_text):
    from tonematch.tone3000 import T3KError

    if not tones:
        raise gr.Error("Search first, then pick rows to download.")
    try:
        picks = [int(s) for s in rows_text.replace(",", " ").split()]
    except ValueError:
        raise gr.Error("Row numbers, please — e.g.  1 3 7")
    bad = [p for p in picks if not 1 <= p <= len(tones)]
    if bad:
        raise gr.Error(f"Row(s) out of range: {bad}")
    client = _get_t3k()
    downloaded = []
    try:
        for p in picks:
            downloaded.extend(client.download_tone(tones[p - 1]["id"], T3K_CACHE))
    except T3KError as e:
        raise gr.Error(str(e))
    return f"⬇️ Downloaded {len(downloaded)} .nam file(s) to `{T3K_CACHE}` — folder set below.", T3K_CACHE


# ---------------------------------------------------------------------------
# Match job
# ---------------------------------------------------------------------------


def _run_match_job(progress_cb, target, di, sources, demix, stem, gain_lo, gain_hi,
                   refine_top, render_top, limit, device, preview_s):
    """Executed by the JobManager worker thread. Returns a result dict."""
    stem_path = None
    if demix:
        from tonematch.stems import extract_stem

        stem_dir = tempfile.mkdtemp(prefix="tonematch_stems_")
        target = extract_stem(target, stem_dir, stem=stem,
                              progress_cb=lambda f, m: progress_cb(0.08 * f, f"Demucs: {m}"))
        stem_path = target

    progress_cb(0.09, f"loading NAM captures ({device})")
    captures, load_errors = [], []
    for s in sources:
        captures.extend(load_captures(s, limit=int(limit) if limit else None,
                                      errors_out=load_errors, device=device))
    seen, uniq = set(), []
    for c in captures:
        if c.path not in seen:
            seen.add(c.path)
            uniq.append(c)
    captures = uniq[: int(limit)] if limit else uniq
    if not captures:
        detail = "; ".join(f"{os.path.basename(p)} - {m}" for p, m in load_errors[:3])
        raise ValueError(f"No NAM captures could be loaded. {detail}")

    out_dir = tempfile.mkdtemp(prefix="tonematch_")
    result = run_match(
        target, di, captures, out_dir,
        gain_range_db=(float(gain_lo), float(gain_hi)),
        refine_top=max(int(refine_top), int(render_top)),
        render_top=int(render_top),
        preview_s=float(preview_s),
        progress_cb=lambda f, m: progress_cb(0.1 + 0.9 * f, f"{m} ({len(captures)} captures)"),
    )
    return {"output": result, "stem_path": stem_path}


def submit_match(target, di, models_dir, model_files, library_name, demix, stem,
                 gain_lo, gain_hi, refine_top, render_top, limit, device, preview_s):
    if target is None or di is None:
        raise gr.Error("Please provide both a target recording and a DI track.")

    sources = []
    label_src = []
    if library_name and library_name != "(none)":
        paths = libmod.library_paths(library_name)
        if paths:
            sources.append(paths)
            label_src.append(f"lib:{library_name}")
    if models_dir and os.path.isdir(models_dir.strip()):
        sources.append(models_dir.strip())
        label_src.append(os.path.basename(models_dir.strip()))
    if model_files:
        sources.append([f.name if hasattr(f, "name") else f for f in model_files])
        label_src.append(f"{len(model_files)} uploaded")
    if not sources:
        raise gr.Error("Pick a saved library, a folder of .nam files, or upload some.")

    _save_session(models_dir=models_dir, gain_lo=gain_lo, gain_hi=gain_hi,
                  refine_top=refine_top, render_top=render_top, device=device,
                  preview_s=preview_s, library=library_name)

    dev = resolve_device(device)
    label = f"{os.path.basename(str(target))[:24]} vs {', '.join(label_src)}"
    job_id = jm.submit(
        label, _run_match_job,
        target=target, di=di, sources=sources, demix=bool(demix), stem=stem,
        gain_lo=gain_lo, gain_hi=gain_hi, refine_top=refine_top,
        render_top=render_top, limit=limit, device=dev, preview_s=preview_s,
    )
    return f"🚀 Queued as **job #{job_id}** ({dev}). Watch the **Jobs** tab; audition in **Results** when done."


# ---------------------------------------------------------------------------
# Jobs tab
# ---------------------------------------------------------------------------


def refresh_jobs():
    rows = [[j["id"], j["label"], j["status"], j["progress"], j["message"], j["runtime"]]
            for j in jm.snapshot()]
    done = [f"#{j.id} {j.label}" for j in jm.completed()]
    return rows, gr.update(choices=done)


def cancel_job(job_id):
    try:
        ok = jm.cancel(int(job_id))
    except (TypeError, ValueError):
        raise gr.Error("Enter the numeric job id to cancel.")
    return "Canceled." if ok else "Job is not queued (already running, done, or unknown)."


# ---------------------------------------------------------------------------
# Results tab
# ---------------------------------------------------------------------------

MATERIALS = ["NAM copy", "Settings", "Match IR", "Gentle IR", "IR render",
             "EQ render", "Hybrid render", "Report", "Plot", "Extracted stem"]


def _job_from_choice(choice):
    if not choice:
        return None
    try:
        return jm.get(int(choice.split()[0].lstrip("#")))
    except (ValueError, IndexError):
        return None


def _rig_label(r):
    ts = r.get("tone_stack") or {}
    eq = f", EQ B{ts['bass']:g}/M{ts['middle']:g}/T{ts['treble']:g}" if ts else ""
    return f"#{r['rank']} {r['name']} (gain {r['input_gain_db']:+.1f} dB{eq})"


def show_result(choice):
    job = _job_from_choice(choice)
    empty = (gr.update(), None, None, None, None, None, gr.update(choices=[]), None)
    if job is None or job.status != "done":
        return ("Select a completed job.",) + empty[1:]
    result = job.result["output"]
    stem_path = job.result.get("stem_path")

    best = result.report["best_model"]
    ts = result.renders[0]["tone_stack"]
    summary = (
        f"### 🏆 Job #{job.id}: **{best['name']}**\n"
        f"- Input gain: **{best['input_gain_db']:+.1f} dB**\n"
        f"- Plugin EQ suggestion: **Bass {ts['bass']:g} · Middle {ts['middle']:g} · Treble {ts['treble']:g}**\n"
        f"- Model file: `{best['file']}`\n\n"
        f"**{result.report['how_to_use']}**"
    )
    if len(result.renders) > 1:
        summary += "\n\n### Rendered rigs\n" + "\n".join(f"- {_rig_label(r)}" for r in result.renders)

    rig_choices = [_rig_label(r) for r in result.renders]
    return (
        summary,
        result.target_ref_path,
        result.render_path,
        result.renders[0]["tone_stack"]["render"],
        result.renders[0]["hybrid"]["render"],
        result.plot_path,
        gr.update(choices=rig_choices, value=rig_choices),
        None,
    )


def export_zip(choice, rig_labels, materials):
    job = _job_from_choice(choice)
    if job is None or job.status != "done":
        raise gr.Error("Select a completed job first.")
    if not materials:
        raise gr.Error("Pick at least one file type to export.")
    result = job.result["output"]
    stem_path = job.result.get("stem_path")
    ranks = {int(l.split()[0].lstrip("#")) for l in (rig_labels or [])}

    paths = []
    for r in result.renders:
        if ranks and r["rank"] not in ranks:
            continue
        m = {
            "NAM copy": r.get("nam_copy"),
            "Settings": r.get("settings_txt"),
            "Match IR": r.get("ir"),
            "Gentle IR": r.get("hybrid", {}).get("gentle_ir"),
            "IR render": r.get("render"),
            "EQ render": r.get("tone_stack", {}).get("render"),
            "Hybrid render": r.get("hybrid", {}).get("render"),
        }
        paths.extend(p for k, p in m.items() if k in materials and p)
    if "Report" in materials:
        paths.append(result.report_path)
    if "Plot" in materials and result.plot_path:
        paths.append(result.plot_path)
    if "Extracted stem" in materials and stem_path:
        paths.append(stem_path)
    paths = [p for p in dict.fromkeys(paths) if p and os.path.isfile(p)]
    if not paths:
        raise gr.Error("Nothing matched your selection.")

    zip_path = os.path.join(os.path.dirname(result.report_path), f"nam_eq_matcher_job{job.id}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, arcname=os.path.basename(p))
    return zip_path


# ---------------------------------------------------------------------------
# Library tab
# ---------------------------------------------------------------------------


def _entries_rows(entries):
    return [[i + 1, e["name"], e["make"], e["model"], e["tone_type"] or "-",
             e["architecture"], os.path.basename(e["path"])] for i, e in enumerate(entries)]


def lib_scan(folder):
    folder = (folder or "").strip()
    if not folder or not os.path.isdir(folder):
        raise gr.Error("Enter a valid folder path to scan.")
    entries, errors = libmod.scan_folder(folder)
    msg = f"Found {len(entries)} captures."
    if errors:
        msg += f" ({len(errors)} unreadable: {'; '.join(errors[:3])}...)"
    return entries, entries, _entries_rows(entries), msg


def lib_filter(entries, text, tone_type, gear_type):
    filtered = libmod.filter_entries(entries or [], text, tone_type, gear_type)
    return filtered, _entries_rows(filtered), f"{len(filtered)} / {len(entries or [])} captures match."


def lib_save(filtered, name):
    if not filtered:
        raise gr.Error("Scan and filter first — the current filtered set is what gets saved.")
    try:
        libmod.save_library(name, [e["path"] for e in filtered])
    except ValueError as e:
        raise gr.Error(str(e))
    libs = ["(none)"] + sorted(libmod.load_libraries())
    return (f"💾 Saved library '{name}' with {len(filtered)} captures.",
            gr.update(choices=libs, value=name), gr.update(choices=libs))


def lib_delete(name):
    if not name or name == "(none)":
        raise gr.Error("Pick a library to delete.")
    libmod.delete_library(name)
    libs = ["(none)"] + sorted(libmod.load_libraries())
    return (f"🗑️ Deleted library '{name}'.",
            gr.update(choices=libs, value="(none)"), gr.update(choices=libs, value="(none)"))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_libs = ["(none)"] + sorted(libmod.load_libraries())

with gr.Blocks(title="NAM EQ Matcher") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        # ------------------------------------------------------------- Match
        with gr.Tab("Match"):
            with gr.Row():
                with gr.Column():
                    target_in = gr.Audio(label="Target recording (isolated guitar or full mix)", type="filepath")
                    di_in = gr.Audio(label="Your DI track (clean)", type="filepath")
                    library_in = gr.Dropdown(_libs, value=_SESSION.get("library", "(none)"),
                                             label="Saved library (Library tab)")
                    models_dir_in = gr.Textbox(label="...or NAM models folder",
                                               value=_SESSION.get("models_dir", ""),
                                               placeholder=r"e.g. C:\Users\you\Documents\NAM models")
                    model_files_in = gr.Files(label="...or upload .nam files", file_types=[".nam"])
                    with gr.Accordion("🔍 Search TONE3000 (download only what you need)", open=False):
                        t3k_key_in = gr.Textbox(label="Publishable API key", type="password",
                                                placeholder="t3k_pub_...  (tone3000.com → Settings → API Keys)")
                        t3k_connect_btn = gr.Button("Connect TONE3000 account")
                        t3k_status = gr.Markdown()
                        with gr.Row():
                            t3k_query_in = gr.Textbox(label="Search", placeholder="e.g. british crunch...", scale=3)
                            t3k_gear_in = gr.Dropdown(["any", "amp", "full-rig", "pedal", "outboard"],
                                                      value="any", label="Gear", scale=1)
                            t3k_sort_in = gr.Dropdown(["downloads-all-time", "trending", "best-match", "newest"],
                                                      value="downloads-all-time", label="Sort", scale=1)
                        t3k_amps_in = gr.Textbox(label="Amps you're looking for (optional, comma-separated)",
                                                 placeholder="e.g. Marshall JCM800, Friedman BE-100, 5150")
                        t3k_search_btn = gr.Button("Search catalog (no downloads)")
                        t3k_results = gr.Dataframe(
                            headers=["#", "ID", "Title", "Make", "Matched on", "Gear", "Models", "Downloads"],
                            label="Results (metadata only)", interactive=False)
                        t3k_tones_state = gr.State([])
                        with gr.Row():
                            t3k_rows_in = gr.Textbox(label="Rows to download", placeholder="e.g. 1 3 7", scale=3)
                            t3k_dl_btn = gr.Button("Download selected", scale=1)
                    with gr.Row():
                        demix_in = gr.Checkbox(label="Target is a full mix — extract guitar stem first (Demucs)")
                        stem_in = gr.Dropdown(["guitar", "other", "guitar+other"], value="guitar",
                                              label="Stem", scale=0)
                    with gr.Accordion("Advanced", open=False):
                        gain_lo = gr.Slider(-24, 0, value=_SESSION.get("gain_lo", -12), step=1,
                                            label="Input gain search min (dB)")
                        gain_hi = gr.Slider(0, 24, value=_SESSION.get("gain_hi", 12), step=1,
                                            label="Input gain search max (dB)")
                        refine_top = gr.Slider(1, 10, value=_SESSION.get("refine_top", 5), step=1,
                                               label="Captures to refine (stage 2)")
                        render_top = gr.Slider(1, 10, value=_SESSION.get("render_top", 1), step=1,
                                               label="Top rigs to render — each gets IR + EQ + hybrid")
                        limit = gr.Number(value=0, precision=0, label="Max captures to load (0 = all)")
                        device_in = gr.Dropdown(["auto", "cpu", "cuda"], value=_SESSION.get("device", "auto"),
                                                label="Processing device (auto = GPU if available)")
                        preview_s_in = gr.Slider(0, 120, value=_SESSION.get("preview_s", 30), step=5,
                                                 label="Render length (s) — loudest DI section; 0 = full DI (slow)")
                    go = gr.Button("Queue tone match", variant="primary")
                    submit_status = gr.Markdown()

        # ----------------------------------------------------------- Library
        with gr.Tab("Library"):
            gr.Markdown("Scan a folder of `.nam` captures, filter by metadata, and save the "
                        "filtered set as a **named library** you can select in the Match tab.")
            with gr.Row():
                lib_folder_in = gr.Textbox(label="Folder to scan (recursive)",
                                           placeholder=r"e.g. H:\NAM captures", scale=3)
                lib_scan_btn = gr.Button("Scan", scale=1)
            lib_status = gr.Markdown()
            with gr.Row():
                lib_text_in = gr.Textbox(label="Filter text (name/make/model)", scale=2)
                lib_tone_in = gr.Dropdown(list(libmod.TONE_TYPES), value="any", label="Tone type", scale=1)
                lib_gear_in = gr.Dropdown(list(libmod.GEAR_TYPES), value="any", label="Gear type", scale=1)
                lib_filter_btn = gr.Button("Apply filter", scale=1)
            lib_table = gr.Dataframe(
                headers=["#", "Name", "Make", "Model", "Tone", "Arch", "File"],
                label="Captures (filtered view is what gets saved)", interactive=False)
            with gr.Row():
                lib_name_in = gr.Textbox(label="Save filtered set as library", placeholder="e.g. Marshall hi-gain", scale=2)
                lib_save_btn = gr.Button("Save library", scale=1)
            with gr.Row():
                lib_manage_dd = gr.Dropdown(_libs, value="(none)", label="Existing libraries", scale=2)
                lib_delete_btn = gr.Button("Delete selected library", scale=1)
            lib_entries_state = gr.State([])
            lib_filtered_state = gr.State([])

        # -------------------------------------------------------------- Jobs
        with gr.Tab("Jobs"):
            gr.Markdown("Queued and running matches. The table refreshes automatically every 2 s.")
            jobs_table = gr.Dataframe(
                headers=["#", "Label", "Status", "Progress", "Message", "Runtime"],
                label="Job queue (newest first)", interactive=False)
            with gr.Row():
                cancel_id_in = gr.Textbox(label="Cancel queued job (id)", scale=2)
                cancel_btn = gr.Button("Cancel", scale=1)
            cancel_status = gr.Markdown()

        # ----------------------------------------------------------- Results
        with gr.Tab("Results"):
            result_dd = gr.Dropdown([], label="Completed job")
            summary_out = gr.Markdown()
            target_audio_out = gr.Audio(label="A: Target (reference)")
            render_audio_out = gr.Audio(label="B: NAM + match IR (#1, closest match)")
            eq_audio_out = gr.Audio(label="C: NAM + plugin EQ only (#1, most natural)")
            hybrid_audio_out = gr.Audio(label="D: NAM + plugin EQ + gentle IR (#1, recommended)")
            plot_out = gr.Image(label="Spectrum match")
            gr.Markdown("### Export selected files as ZIP")
            rig_select = gr.CheckboxGroup([], label="Rigs")
            material_select = gr.CheckboxGroup(
                MATERIALS, value=["NAM copy", "Settings", "Match IR", "Gentle IR", "Hybrid render"],
                label="File types")
            zip_btn = gr.Button("Create ZIP")
            zip_out = gr.File(label="Download ZIP")

    # ------------------------------------------------------------ wiring
    t3k_connect_btn.click(t3k_connect, inputs=[t3k_key_in], outputs=[t3k_status])
    t3k_search_btn.click(t3k_search, inputs=[t3k_query_in, t3k_amps_in, t3k_gear_in, t3k_sort_in],
                         outputs=[t3k_results, t3k_tones_state, t3k_status])
    t3k_dl_btn.click(t3k_download, inputs=[t3k_tones_state, t3k_rows_in],
                     outputs=[t3k_status, models_dir_in])

    go.click(
        submit_match,
        inputs=[target_in, di_in, models_dir_in, model_files_in, library_in, demix_in, stem_in,
                gain_lo, gain_hi, refine_top, render_top, limit, device_in, preview_s_in],
        outputs=[submit_status],
    )

    lib_scan_btn.click(lib_scan, inputs=[lib_folder_in],
                       outputs=[lib_entries_state, lib_filtered_state, lib_table, lib_status])
    lib_filter_btn.click(lib_filter, inputs=[lib_entries_state, lib_text_in, lib_tone_in, lib_gear_in],
                         outputs=[lib_filtered_state, lib_table, lib_status])
    lib_save_btn.click(lib_save, inputs=[lib_filtered_state, lib_name_in],
                       outputs=[lib_status, library_in, lib_manage_dd])
    lib_delete_btn.click(lib_delete, inputs=[lib_manage_dd],
                         outputs=[lib_status, library_in, lib_manage_dd])

    timer = gr.Timer(2)
    timer.tick(refresh_jobs, outputs=[jobs_table, result_dd])
    cancel_btn.click(cancel_job, inputs=[cancel_id_in], outputs=[cancel_status])

    result_dd.change(
        show_result, inputs=[result_dd],
        outputs=[summary_out, target_audio_out, render_audio_out, eq_audio_out,
                 hybrid_audio_out, plot_out, rig_select, zip_out],
    )
    zip_btn.click(export_zip, inputs=[result_dd, rig_select, material_select], outputs=[zip_out])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
