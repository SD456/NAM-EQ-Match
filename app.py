"""NAM EQ Matcher GUI (Gradio).

Run:  python app.py   then open http://127.0.0.1:7860
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile

import gradio as gr

from tonematch.job_manager import JobManager, _run_match_core
from tonematch.nam_backend import load_captures
from tonematch.pipeline import run_match
from tonematch.library import (
    scan_folder,
    scan_for_dataframe,
    save_library,
    load_library_file_paths,
    load_library_path,
    list_libraries,
    delete_library,
    library_file_count,
)

T3K_CACHE = os.path.abspath("t3k_cache")
DEFAULT_DOWNLOAD_DIR = os.path.join(os.environ.get("USERPROFILE", "C:\\"), "Downloads")
SESSION_PATH = os.path.abspath("last_session.json")

DESCRIPTION = """
# NAM EQ Matcher — clone a guitar tone from a recording

1. **Target**: a recording with the tone you want — an isolated guitar track, or a **full mix**
   (tick *Extract guitar stem* and Demucs will demix it first).
2. **DI**: your own clean (unamped) guitar take.
3. **NAM library**: a folder of `.nam` captures, or a **curated library** saved from the Library tab,
   or search **TONE3000** below and download only the shortlist you want to try.

NAM EQ Matcher reamps your DI through every capture, searches the input gain that matches the
target's **saturation & compression character**, then designs a **match IR** that corrects
the remaining EQ/cab difference. You get the winning model + gain setting + IR to load
straight into the NAM plugin.

Queue multiple jobs — tweak settings and hit **Run** again to queue another. Check the **Jobs** tab for progress, then **Results** to audition and download.
"""

# ---------------------------------------------------------------------------
# Session persistence (minimal — last browsed directories only)
# ---------------------------------------------------------------------------

_SESSION_KEYS = {"last_models_dir", "last_download_dir", "last_scan_folder", "last_materials",
                 "last_device", "last_preview_s"}


def _load_session() -> dict:
    if os.path.isfile(SESSION_PATH):
        try:
            with open(SESSION_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {k: v for k, v in data.items() if k in _SESSION_KEYS}
        except Exception:
            pass
    return {}


def _save_session(**kwargs) -> None:
    try:
        existing = _load_session()
        existing.update(kwargs)
        existing = {k: v for k, v in existing.items() if k in _SESSION_KEYS}
        tmp = SESSION_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        os.replace(tmp, SESSION_PATH)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Singleton job manager
# ---------------------------------------------------------------------------

_job_manager = JobManager()

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
        return "Connected to TONE3000."
    except T3KError as e:
        return f"Error: {e}"


def t3k_search(query, amps_text, gear, sort):
    from tonematch.tone3000 import T3KError
    client = _get_t3k()
    amps = [a for a in (amps_text or "").split(",") if a.strip()]
    try:
        tones = client.search_tones_refined(
            query or "", amps=amps, gear=gear, sort=sort, page_size=30
        )
    except T3KError as e:
        raise gr.Error(str(e))
    rows = [
        [
            i + 1,
            t["id"],
            t["title"],
            ", ".join(m["name"] for m in t.get("makes", [])) or "-",
            t.get("_matched_on", "-"),
            t["gear"],
            t["models_count"],
            t["downloads_count"],
        ]
        for i, t in enumerate(tones)
    ]
    hint = f" (refined by: {', '.join(a.strip() for a in amps)})" if amps else ""
    status = f"{len(tones)} NAM tones found{hint}. Nothing downloaded yet — pick rows below."
    return rows, tones, status


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
    status = f"Downloaded {len(downloaded)} .nam file(s) to `{T3K_CACHE}` — folder set below."
    return status, T3K_CACHE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rig_label(r):
    ts = r.get("tone_stack") or {}
    eq = f", EQ B{ts['bass']:g}/M{ts['middle']:g}/T{ts['treble']:g}" if ts else ""
    return f"#{r['rank']} {r['name']} (gain {r['input_gain_db']:+.1f} dB{eq})"


def _browse_for_folder():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory()
        root.destroy()
        return path if path else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Results-panel builder
# ---------------------------------------------------------------------------

ALL_MATERIALS = ["NAM model", "Settings (.txt)", "Match IR",
                 "Plugin EQ render", "IR-matched render",
                 "Hybrid render", "Gentle IR",
                 "Report (report.json)", "Target reference",
                 "Spectrum plot", "Demuxed stem"]


def _build_results(job):
    """Return the output values for the Results tab from a Job."""
    result = job.result
    best = result.report["best_model"]
    best_ts = result.renders[0]["tone_stack"]

    rows = [
        [i + 1, r.name, f"{r.gain_db:+.1f}", f"{r.score:.4f}",
         f"{r.nl_distance:.4f}", f"{r.eq_penalty:.4f}"]
        for i, r in enumerate(result.ranked[:15])
    ]

    summary = (
        f"### Best match: **{best['name']}**\n"
        f"- Input gain: **{best['input_gain_db']:+.1f} dB**\n"
        f"- Model file: `{best['file']}`\n\n"
        f"### Plugin EQ Settings:\n"
        f"- **Bass**: {best_ts['bass']:g} (Gain: {best_ts['bass_gain_db']:+.1f} dB)\n"
        f"- **Middle**: {best_ts['middle']:g} (Gain: {best_ts['middle_gain_db']:+.1f} dB)\n"
        f"- **Treble**: {best_ts['treble']:g} (Gain: {best_ts['treble_gain_db']:+.1f} dB)\n\n"
        f"**How to use:**\n{result.report['how_to_use']}"
    )
    if len(result.renders) > 1:
        summary += "\n\n### Rendered rigs\n" + "\n".join(
            f"- {_rig_label(r)}" for r in result.renders
        )

    dl_mapping = {}
    rig_choices = []
    rig_values = []
    for r in result.renders:
        rank = r["rank"]
        label = _rig_label(r)
        rig_choices.append((label, rank))
        rig_values.append(rank)
        if r.get("nam_copy"):
            dl_mapping[(rank, "NAM model")] = r["nam_copy"]
        dl_mapping[(rank, "Settings (.txt)")] = r["settings_txt"]
        dl_mapping[(rank, "Match IR")] = r["ir"]
        dl_mapping[(rank, "Plugin EQ render")] = r["tone_stack"]["render"]
        dl_mapping[(rank, "IR-matched render")] = r["render"]
        dl_mapping[(rank, "Hybrid render")] = r["hybrid"]["render"]
        dl_mapping[(rank, "Gentle IR")] = r["hybrid"]["gentle_ir"]
        if r.get("plot"):
            dl_mapping[(rank, "Spectrum plot")] = r["plot"]

    dl_mapping[(None, "Report (report.json)")] = result.report_path
    dl_mapping[(None, "Target reference")] = result.target_ref_path
    if result.plot_path:
        dl_mapping[(None, "Spectrum plot")] = result.plot_path
    if job.stem_path:
        dl_mapping[(None, "Demuxed stem")] = job.stem_path

    first_rank = rig_values[0] if rig_values else None

    plot_file = result.renders[0].get("plot") or result.plot_path
    for r in result.renders:
        r["plot"] = plot_file

    return (
        summary,
        rows,
        result.target_ref_path,
        result.renders[0]["render"],
        result.renders[0]["tone_stack"]["render"],
        result.renders[0]["hybrid"]["render"],
        result.renders[0].get("plot") or result.plot_path,
        result.renders,
        dl_mapping,
        gr.update(choices=rig_choices, value=first_rank),
        gr.update(choices=ALL_MATERIALS,
                  value=["NAM model", "Settings (.txt)", "Match IR"], visible=True),
        gr.update(choices=rig_choices, value=rig_values, visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
    )


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _dl_filtered(mapping, rig_ranks, materials):
    if not mapping or not materials:
        return []
    ranks = set(rig_ranks) if rig_ranks else set()
    paths = []
    for (rank, mat), p in mapping.items():
        if mat in materials and (rank is None or rank in ranks) and p and os.path.isfile(p):
            paths.append(p)
    return paths


def _dl_selected_files(mapping, rig_ranks, materials, download_folder):
    paths = _dl_filtered(mapping, rig_ranks, materials)
    if not paths:
        return []
    os.makedirs(download_folder, exist_ok=True)
    zip_name = f"tone_match_selected_{int(__import__('time').time())}.zip"
    zip_path = os.path.join(download_folder, zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for p in paths:
            zipf.write(p, os.path.basename(p))
    return [zip_path]


def _dl_all_files(mapping, download_folder):
    if not mapping:
        return []
    os.makedirs(download_folder, exist_ok=True)
    zip_name = f"tone_match_all_{int(__import__('time').time())}.zip"
    zip_path = os.path.join(download_folder, zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for (_rank, _mat), p in mapping.items():
            if p and os.path.isfile(p):
                zipf.write(p, os.path.basename(p))
    return [zip_path]


# ---------------------------------------------------------------------------
# Library helpers (Gradio handlers)
# ---------------------------------------------------------------------------

def _on_scan_folder(folder):
    if not folder:
        raise gr.Error("Select a folder to scan.")
    if not os.path.isdir(folder):
        raise gr.Error(f"Folder not found: {folder}")
    items = scan_folder(folder)
    if not items:
        raise gr.Error("No .nam files found in that folder.")
    df_rows = scan_for_dataframe(items)
    all_paths = [i["filepath"] for i in items]
    brands = sorted({i["brand"] for i in items})
    tones = sorted({i["tone_category"] for i in items})
    instruments = sorted({i["instrument"] for i in items})
    types = sorted({i["type"] for i in items})
    status = f"Loaded {len(items)} profiles"
    return (
        items, df_rows,
        gr.update(choices=all_paths, value=[]), [],
        status,
        gr.update(choices=brands, value=brands),
        gr.update(choices=tones, value=tones),
        gr.update(choices=instruments, value=instruments),
        gr.update(choices=types, value=types),
    )


def _on_lib_filter(search, brands, tones, instruments, types, all_items, selected_paths):
    if not all_items:
        return [], [], "No data loaded"
    if isinstance(selected_paths, list):
        selected_paths = set(selected_paths)
    brands = list(brands) if brands else []
    tones = list(tones) if tones else []
    instruments = list(instruments) if instruments else []
    types = list(types) if types else []
    def match(item):
        if search:
            haystack = f"{item['display_name']} {item['brand']} {item['tone_category']} {item['filename']}".lower()
            if search.lower() not in haystack:
                return False
        if brands and item["brand"] not in brands:
            return False
        if tones and item["tone_category"] not in tones:
            return False
        if instruments and item["instrument"] not in instruments:
            return False
        if types and item["type"] not in types:
            return False
        return True
    filtered = [i for i in all_items if match(i)]
    df_rows = scan_for_dataframe(filtered)
    filtered_paths = [i["filepath"] for i in filtered]
    n = len(filtered)
    total = len(all_items)
    sel = len(selected_paths)
    status = f"Showing {n} of {total} | {sel} selected"
    return df_rows, gr.update(choices=filtered_paths, value=sorted(selected_paths)), status


def _on_select_all(all_items, selected_paths, brands, tones, instruments, types, search):
    if not all_items:
        return [], "No data"
    if isinstance(selected_paths, list):
        selected_paths = set(selected_paths)
    brands = list(brands) if brands else []
    tones = list(tones) if tones else []
    instruments = list(instruments) if instruments else []
    types = list(types) if types else []
    def match(item):
        if search:
            haystack = f"{item['display_name']} {item['brand']} {item['tone_category']} {item['filename']}".lower()
            if search.lower() not in haystack:
                return False
        if brands and item["brand"] not in brands:
            return False
        if tones and item["tone_category"] not in tones:
            return False
        if instruments and item["instrument"] not in instruments:
            return False
        if types and item["type"] not in types:
            return False
        return True
    new_paths = set(selected_paths) | {i["filepath"] for i in all_items if match(i)}
    new_list = sorted(new_paths)
    return new_list, f"Selected {len(new_list)} files"


def _on_clear_selection():
    return [], "Selection cleared"


def _on_save_library(name, selected_paths):
    if not name or not name.strip():
        raise gr.Error("Enter a library name.")
    if isinstance(selected_paths, list):
        selected_paths_set = set(selected_paths)
    else:
        selected_paths_set = set(selected_paths)
    if not selected_paths_set:
        raise gr.Error("No files selected.")
    sanitized = save_library(name.strip(), sorted(selected_paths_set))
    libs = list_libraries()
    choices = [(f"{n} ({library_file_count(n)} files)", n) for n in libs]
    count = len(selected_paths_set)
    return gr.update(choices=choices), f"Saved '{sanitized}' ({count} files)"


def _on_delete_checked(checked_names):
    if not checked_names:
        raise gr.Error("Check libraries to delete.")
    for name in checked_names:
        delete_library(name)
    libs = list_libraries()
    choices = [(f"{n} ({library_file_count(n)} files)", n) for n in libs]
    return gr.update(choices=choices, value=[]), f"Deleted {len(checked_names)} library/ies."


def _list_lib_choices():
    libs = list_libraries()
    return [(f"{n} ({library_file_count(n)} files)", n) for n in libs]


# ---------------------------------------------------------------------------
# Job selector helpers
# ---------------------------------------------------------------------------

def _snap_to_rows(snap):
    return [[j["label"], j["status"], f"{j['progress']:.0%}", j["message"]] for j in snap]


def _done_job_choices(snap):
    return [(f"{j['label']} ({j['status']})", j["id"]) for j in snap if j["status"] == "done"]


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------



with gr.Blocks(title="NAM EQ Matcher") as demo:
    gr.Markdown(DESCRIPTION)

    _scroll_js = r"""
    (function() {
        /* Inject dark-theme scrollbar styles */
        var style = document.createElement('style');
        style.textContent = [
            '::-webkit-scrollbar { width: 8px; height: 8px; }',
            '::-webkit-scrollbar-track { background: #1e1e2e; }',
            '::-webkit-scrollbar-thumb { background: #585b70; border-radius: 4px; }',
            '::-webkit-scrollbar-thumb:hover { background: #7f849c; }',
            'ul.options { scrollbar-width: thin; scrollbar-color: #585b70 #1e1e2e; }',
        ].join('\n');
        document.head.appendChild(style);

        function tick() {
            document.querySelectorAll('ul.options').forEach(function(el) {
                el.style.removeProperty('bottom');
                el.style.maxHeight = '250px';
                el.style.overflowY = 'auto';
            });
            document.querySelectorAll('.filter-scroll .wrap, .lib-select .wrap').forEach(function(el) {
                el.style.maxHeight = '120px';
                el.style.overflowY = 'auto';
            });
            document.querySelectorAll('.check-scroll').forEach(function(el) {
                el.style.maxHeight = '150px';
                el.style.overflowY = 'auto';
            });
        }
        tick();
        new MutationObserver(function(mutations) {
            tick();
        }).observe(document.body, {childList: true, subtree: true});

        /* Make entire dropdown clickable via pointerdown (works with Gradio's event handling) */
        document.addEventListener('pointerdown', function(e) {
            var el = e.target;
            while (el && el !== document.body) {
                if (el.classList && el.classList.contains('gradio-dropdown')) break;
                el = el.parentElement;
            }
            if (!el || el === document.body) return;
            if (e.target.tagName === 'INPUT') return;
            if (e.target.closest && e.target.closest('ul.options')) return;
            var input = el.querySelector('input');
            if (input) { input.focus(); }
        }, true);
    })();
    """
    gr.HTML(value="&nbsp;", elem_id="scroll-inject", visible=True, container=False, padding=False,
            min_width=0, scale=0, js_on_load=_scroll_js)

    # Global states
    jobs_snapshot_state = gr.State([])

    # ==================================================================
    # Tabs
    # ==================================================================

    with gr.Tabs(selected=1):
        # ==============================================================
        # Library tab
        # ==============================================================
        with gr.Tab("Library"):
            lib_all_items = gr.State([])
            lib_selected_paths = gr.State([])

            lib_folder_in = gr.Textbox(label="Folder to scan")
            with gr.Row():
                lib_browse_btn = gr.Button("Browse...", scale=1)
                lib_scan_btn = gr.Button("Scan folder", variant="primary", scale=1)

            with gr.Row():
                lib_search_in = gr.Textbox(label="Search", placeholder="name, brand...", scale=2)
                lib_brand_filter = gr.Dropdown(label="Brand", multiselect=True, choices=[], scale=1, elem_classes="filter-scroll")
                lib_tone_filter = gr.Dropdown(label="Tone", multiselect=True, choices=[], scale=1, elem_classes="filter-scroll")
                lib_instr_filter = gr.Dropdown(label="Instrument", multiselect=True, choices=[], scale=1, elem_classes="filter-scroll")
                lib_type_filter = gr.Dropdown(label="Type", multiselect=True, choices=[], scale=1, elem_classes="filter-scroll")

            lib_status = gr.Markdown("No data loaded")

            with gr.Row():
                lib_select_all_btn = gr.Button("Select All Visible")
                lib_clear_btn = gr.Button("Clear Selection")

            lib_dataframe = gr.Dataframe(
                headers=["Name", "Brand", "Tone", "Instrument", "Type", "File"],
                label="Files", interactive=False,
            )

            lib_select_in = gr.Dropdown(
                label="Selected files (check to include)",
                multiselect=True, choices=[], value=[], scale=1, elem_classes="lib-select",
            )

            with gr.Row():
                lib_name_in = gr.Textbox(label="Library name", placeholder="e.g. 5150 Collection", scale=2)
                lib_lib_check = gr.CheckboxGroup(label="Saved libraries", choices=[], value=[], elem_classes="check-scroll", scale=1)
            with gr.Row():
                lib_save_btn = gr.Button("Save Library", variant="primary", scale=1)
                lib_delete_checked_btn = gr.Button("Delete Library", variant="secondary", scale=1)
            lib_lib_status = gr.Markdown()

        # ==============================================================
        # Workflow tab
        # ==============================================================
        with gr.Tab("Workflow", id=1):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        target_in = gr.Audio(label="Target recording", type="filepath")
                        di_in = gr.Audio(label="Your DI track", type="filepath")

                    source_mode = gr.Radio(["Library", "Folder"], value="Library", label="Source type")

                    browse_btn = gr.Button(icon="📁", size="sm", min_width=32)
                    models_dir_in = gr.Textbox(
                        label="NAM models folder",
                        placeholder=r"e.g. C:\Users\you\Documents\NAM models",
                        buttons=[browse_btn],
                        visible=False,
                    )

                    wf_lib_dd = gr.Dropdown(
                        label="Saved library", choices=[], value=None,
                    )

                    with gr.Accordion("Search TONE3000", open=False):
                        t3k_key_in = gr.Textbox(
                            label="Publishable API key",
                            placeholder="t3k_pub_... (blank if already connected)",
                            type="password",
                        )
                        t3k_connect_btn = gr.Button("Connect")
                        t3k_status = gr.Markdown()
                        with gr.Row():
                            t3k_query_in = gr.Textbox(label="Search", scale=3)
                            t3k_gear_in = gr.Dropdown(
                                ["any", "amp", "full-rig", "pedal", "outboard"],
                                value="any", label="Gear", scale=1,
                            )
                            t3k_sort_in = gr.Dropdown(
                                ["downloads-all-time", "trending", "best-match", "newest"],
                                value="downloads-all-time", label="Sort", scale=1,
                            )
                        t3k_amps_in = gr.Textbox(
                            label="Amps (comma-separated, optional)",
                            placeholder="Marshall JCM800, 5150...",
                        )
                        t3k_search_btn = gr.Button("Search catalog")
                        t3k_results = gr.Dataframe(
                            headers=["#", "ID", "Title", "Make", "Matched on", "Gear", "Models", "Downloads"],
                            label="Results (metadata only)", interactive=False,
                        )
                        t3k_tones_state = gr.State([])
                        with gr.Row():
                            t3k_rows_in = gr.Textbox(label="Rows to download", placeholder="1 3 7", scale=3)
                            t3k_dl_btn = gr.Button("Download selected", scale=1)

                    with gr.Row():
                        demix_in = gr.Checkbox(label="Full mix — extract guitar stem (Demucs)", value=False)
                        stem_in = gr.Dropdown(["guitar", "other", "guitar+other"], value="guitar", label="Stem", scale=0)

                    with gr.Accordion("Advanced", open=False):
                        gain_lo_in = gr.Slider(-24, 0, value=-12, step=1, label="Gain search min (dB)")
                        gain_hi_in = gr.Slider(0, 24, value=12, step=1, label="Gain search max (dB)")
                        refine_top_in = gr.Slider(1, 10, value=5, step=1, label="Captures to refine")
                        render_top_in = gr.Slider(1, 10, value=1, step=1, label="Rigs to render")
                        limit_in = gr.Number(value=0, precision=0, label="Max captures (0 = all)")
                        device_in = gr.Dropdown(
                            ["auto", "cpu", "cuda"], value="auto",
                            label="Processing device (auto = GPU if available)",
                        )
                        preview_s_in = gr.Slider(
                            0, 120, value=30, step=5,
                            label="Render length (s) — loudest section of DI; 0 = full DI (slow)",
                        )

                    go = gr.Button("Match my tone", variant="primary")
                    run_status = gr.Markdown()

        # ==============================================================
        # Jobs tab
        # ==============================================================
        with gr.Tab("Jobs"):
            jobs_table = gr.Dataframe(
                headers=["Label", "Status", "Progress", "Message"],
                label="Job queue", interactive=False,
            )

        # ==============================================================
        # Results tab
        # ==============================================================
        with gr.Tab("Results"):
            results_job_selector = gr.Dropdown(
                label="Completed job", choices=[], value=None,
            )

            summary_out = gr.Markdown()
            table_out = gr.Dataframe(
                headers=["#", "Capture", "Gain (dB)", "Score", "NL dist", "EQ penalty"],
                label="Ranking", interactive=False,
            )

            renders_state = gr.State([])
            rig_select = gr.Dropdown(label="Rig to audition", choices=[], value=None, interactive=True)

            target_audio_out = gr.Audio(label="A: Target (reference)")
            rig_render_out = gr.Audio(label="B: DI through matched rig + Match IR")
            rig_eq_out = gr.Audio(label="C: DI through matched rig + Plugin EQ")
            rig_hybrid_out = gr.Audio(label="D: DI through matched rig + Plugin EQ + gentle IR (recommended)")
            plot_out = gr.Image(label="Spectrum match")

            dl_mapping_state = gr.State({})
            dl_rig_toggle = gr.CheckboxGroup(
                choices=[], value=[], label="Rigs to include", visible=False, elem_classes="check-scroll",
            )
            dl_material_toggle = gr.CheckboxGroup(
                label="Materials to include",
                choices=["NAM model", "Settings (.txt)", "Match IR",
                         "Plugin EQ render", "IR-matched render",
                         "Hybrid render", "Gentle IR",
                         "Report (report.json)", "Target reference",
                         "Spectrum plot", "Demuxed stem"],
                value=["NAM model", "Settings (.txt)", "Match IR"],
                elem_classes="check-scroll",
            )
            download_folder_in = gr.Textbox(
                label="Download to folder", value=DEFAULT_DOWNLOAD_DIR,
            )
            dl_browse_btn = gr.Button("Browse...", min_width=80)
            with gr.Row():
                btn_dl_selected = gr.Button("Download Selected (ZIP)", variant="primary", visible=False)
                btn_dl_all = gr.Button("Download All (ZIP)", visible=False)
            files_out = gr.Files(label="Downloads", file_count="multiple", visible=False)

    # ==================================================================
    # Event wiring
    # ==================================================================

    # --- Timer: auto-refresh jobs ---
    def _tick():
        snap = _job_manager.snapshot()
        rows = _snap_to_rows(snap)
        done_choices = _done_job_choices(snap)
        return snap, rows, gr.update(choices=done_choices)

    gr.Timer(2).tick(
        _tick,
        outputs=[jobs_snapshot_state, jobs_table, results_job_selector],
    )

    # --- TONE3000 ---
    t3k_connect_btn.click(t3k_connect, inputs=[t3k_key_in], outputs=[t3k_status])
    t3k_search_btn.click(
        t3k_search,
        inputs=[t3k_query_in, t3k_amps_in, t3k_gear_in, t3k_sort_in],
        outputs=[t3k_results, t3k_tones_state, t3k_status],
    )
    t3k_dl_btn.click(
        t3k_download,
        inputs=[t3k_tones_state, t3k_rows_in],
        outputs=[t3k_status, models_dir_in],
    )

    # --- Browse folders ---
    def _browse_and_save_models(val):
        _save_session(last_models_dir=val)
        return val

    def _browse_and_save_download(val):
        _save_session(last_download_dir=val)
        return val

    def _browse_and_save_scan(val):
        _save_session(last_scan_folder=val)
        return val

    browse_btn.click(_browse_for_folder, outputs=[models_dir_in]).then(
        _browse_and_save_models, inputs=[models_dir_in], outputs=[models_dir_in],
    )
    dl_browse_btn.click(_browse_for_folder, outputs=[download_folder_in]).then(
        _browse_and_save_download, inputs=[download_folder_in], outputs=[download_folder_in],
    )
    lib_browse_btn.click(_browse_for_folder, outputs=[lib_folder_in]).then(
        _browse_and_save_scan, inputs=[lib_folder_in], outputs=[lib_folder_in],
    )

    # --- Source mode toggle ---
    def _toggle_source_mode(mode):
        if mode == "Library":
            return gr.update(visible=False), gr.update(visible=True)
        return gr.update(visible=True), gr.update(visible=False)

    source_mode.change(
        _toggle_source_mode,
        inputs=[source_mode],
        outputs=[models_dir_in, wf_lib_dd],
    )

    # --- Library tab ---
    lib_scan_btn.click(
        _on_scan_folder,
        inputs=[lib_folder_in],
        outputs=[lib_all_items, lib_dataframe,
                 lib_select_in, lib_selected_paths,
                 lib_status,
                 lib_brand_filter, lib_tone_filter,
                 lib_instr_filter, lib_type_filter],
        queue=True,
    )

    for filt in [lib_search_in, lib_brand_filter, lib_tone_filter, lib_instr_filter, lib_type_filter]:
        filt.change(
            _on_lib_filter,
            inputs=[lib_search_in, lib_brand_filter, lib_tone_filter,
                    lib_instr_filter, lib_type_filter, lib_all_items, lib_selected_paths],
            outputs=[lib_dataframe, lib_select_in, lib_status],
        )

    def _on_select_sync(selected_values):
        return selected_values, f"{len(selected_values)} selected"

    lib_select_in.change(
        _on_select_sync,
        inputs=[lib_select_in],
        outputs=[lib_selected_paths, lib_status],
    )

    lib_select_all_btn.click(
        _on_select_all,
        inputs=[lib_all_items, lib_selected_paths, lib_brand_filter,
                lib_tone_filter, lib_instr_filter, lib_type_filter, lib_search_in],
        outputs=[lib_selected_paths, lib_status],
    ).then(
        _on_lib_filter,
        inputs=[lib_search_in, lib_brand_filter, lib_tone_filter,
                lib_instr_filter, lib_type_filter, lib_all_items, lib_selected_paths],
        outputs=[lib_dataframe, lib_select_in, lib_status],
    )

    lib_clear_btn.click(
        _on_clear_selection,
        outputs=[lib_selected_paths, lib_status],
    ).then(
        _on_lib_filter,
        inputs=[lib_search_in, lib_brand_filter, lib_tone_filter,
                lib_instr_filter, lib_type_filter, lib_all_items, lib_selected_paths],
        outputs=[lib_dataframe, lib_select_in, lib_status],
    )

    def _sync_lib_dds():
        choices = _list_lib_choices()
        return gr.update(choices=choices)

    lib_save_btn.click(
        _on_save_library,
        inputs=[lib_name_in, lib_selected_paths],
        outputs=[lib_lib_check, lib_lib_status],
    ).then(_sync_lib_dds, outputs=[wf_lib_dd])

    lib_delete_checked_btn.click(
        _on_delete_checked,
        inputs=[lib_lib_check],
        outputs=[lib_lib_check, lib_lib_status],
    ).then(_sync_lib_dds, outputs=[wf_lib_dd])

    # --- Run button ---
    def _on_run(target, di, source_mode, models_dir, library_name, demix, stem,
                gain_lo, gain_hi, refine_top, render_top, limit, device, preview_s):
        if target is None or di is None:
            raise gr.Error("Please provide both a target recording and a DI track.")
        params = {
            "demix": demix, "stem": stem,
            "gain_lo": gain_lo, "gain_hi": gain_hi,
            "refine_top": refine_top, "render_top": render_top,
            "limit": limit,
            "device": device,
            "preview_s": preview_s,
        }
        if source_mode == "Library":
            if not library_name:
                raise gr.Error("Select a library.")
            lib_files = load_library_file_paths(library_name)
            if not lib_files:
                raise gr.Error("Library is empty.")
            params["library_files"] = lib_files
            models_dir_val = ""
            label = f"Library: {library_name}"
        else:
            if not models_dir or not models_dir.strip():
                raise gr.Error("Please provide a folder of .nam files.")
            models_dir_val = models_dir.strip()
            label = os.path.basename(models_dir_val.rstrip("/\\")) or "Match"
        _job_manager.submit(target, di, models_dir_val, params, label=label)
        snap = _job_manager.snapshot()
        return f"**Job queued** — {label}. Check the Jobs tab.", snap, _snap_to_rows(snap)

    go.click(
        _on_run,
        inputs=[target_in, di_in, source_mode, models_dir_in, wf_lib_dd,
                demix_in, stem_in, gain_lo_in, gain_hi_in,
                refine_top_in, render_top_in, limit_in, device_in, preview_s_in],
        outputs=[run_status, jobs_snapshot_state, jobs_table],
    )

    # --- Results tab: load job ---
    def _load_result_job(job_id):
        if not job_id:
            raise gr.Error("Select a completed job.")
        job = _job_manager.jobs.get(job_id)
        if job is None:
            raise gr.Error("Job not found.")
        if job.status != "done" or job.result is None:
            raise gr.Error(f"Job '{job.label}' is {job.status} — no results yet.")
        return _build_results(job)

    results_job_selector.change(
        _load_result_job,
        inputs=[results_job_selector],
        outputs=[summary_out, table_out,
                 target_audio_out, rig_render_out, rig_eq_out, rig_hybrid_out, plot_out,
                 renders_state, dl_mapping_state,
                 rig_select, dl_material_toggle, dl_rig_toggle,
                 btn_dl_selected, btn_dl_all],
    )

    # --- Rig audition dropdown ---
    def _on_rig_change(value, renders):
        if not renders:
            raise gr.Error("No rig selected.")
        if value is None:
            return None, None, None, None
        if isinstance(value, (int, float)):
            r = next((x for x in renders if x["rank"] == int(value)), None)
        else:
            r = next((x for x in renders if _rig_label(x) == str(value)), None)
        if r is None:
            raise gr.Error("Rig not found in results.")
        return r["render"], r["tone_stack"]["render"], r["hybrid"]["render"], r.get("plot")

    rig_select.change(
        _on_rig_change,
        inputs=[rig_select, renders_state],
        outputs=[rig_render_out, rig_eq_out, rig_hybrid_out, plot_out],
    )

    # --- Download toggles ---
    def _on_dl_change(mapping, rig_ranks, materials):
        return _dl_filtered(mapping, rig_ranks, materials)

    dl_rig_toggle.change(
        _on_dl_change,
        inputs=[dl_mapping_state, dl_rig_toggle, dl_material_toggle],
        outputs=[files_out],
    )
    dl_material_toggle.change(
        _on_dl_change,
        inputs=[dl_mapping_state, dl_rig_toggle, dl_material_toggle],
        outputs=[files_out],
    )

    btn_dl_selected.click(
        _dl_selected_files,
        inputs=[dl_mapping_state, dl_rig_toggle, dl_material_toggle, download_folder_in],
        outputs=[files_out],
    )
    btn_dl_all.click(
        _dl_all_files,
        inputs=[dl_mapping_state, download_folder_in],
        outputs=[files_out],
    )


    # --- Load session + init on page load ---
    def _on_load():
        sess = _load_session()
        lib_choices = _list_lib_choices()
        wf_choices = [(c[1], c[1]) for c in lib_choices]
        saved_materials = sess.get("last_materials", None)
        return (
            sess.get("last_models_dir", ""),
            sess.get("last_download_dir", DEFAULT_DOWNLOAD_DIR),
            sess.get("last_scan_folder", ""),
            gr.update(choices=lib_choices),
            gr.update(choices=wf_choices),
            gr.update(value=saved_materials) if saved_materials else gr.update(),
            sess.get("last_device", "auto"),
            sess.get("last_preview_s", 30),
        )

    demo.load(
        _on_load,
        outputs=[models_dir_in, download_folder_in, lib_folder_in,
                 lib_lib_check, wf_lib_dd, dl_material_toggle,
                 device_in, preview_s_in],
    )

    def _save_materials(materials):
        _save_session(last_materials=materials or [])

    dl_material_toggle.change(
        _save_materials,
        inputs=[dl_material_toggle],
        outputs=[],
    )

if __name__ == "__main__":
    demo.launch()
