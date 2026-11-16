"""NAM capture library: scan folders, filter by metadata, save named sets.

.nam files carry useful metadata (gear_make, gear_model, gear_type, tone_type).
Scanning parses each file once and caches results by (path, mtime) so repeat
scans are instant. Named libraries (name -> list of .nam paths) are stored in
~/.tonematch/libraries.json and can be used as the model source for a match.

Inspired by community PR #2 (library curation concept), re-implemented for the
current codebase.
"""

from __future__ import annotations

import json
import os

CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".tonematch")
LIBRARIES_PATH = os.path.join(CONFIG_DIR, "libraries.json")
SCAN_CACHE_PATH = os.path.join(CONFIG_DIR, "scan_cache.json")

TONE_TYPES = ("any", "clean", "overdrive", "crunch", "hi_gain", "fuzz")
GEAR_TYPES = ("any", "amp", "pedal", "pedal_amp", "amp_cab", "amp_pedal_cab", "preamp", "studio")


def _load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------
# Scanning
# ----------------------------------------------------------------------------


def _read_nam_entry(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        config = json.load(fp)
    arch = config.get("architecture", "?")
    meta = config.get("metadata") or {}
    if arch == "SlimmableContainer":
        try:
            from .nam_backend import unwrap_container

            inner = unwrap_container(config)
            arch = f"A2/{inner.get('architecture', '?')}"
            meta = {**(inner.get("metadata") or {}), **{k: v for k, v in meta.items() if v}}
        except Exception:  # noqa: BLE001
            pass
    return {
        "path": os.path.abspath(path),
        "name": meta.get("name") or os.path.splitext(os.path.basename(path))[0],
        "make": meta.get("gear_make") or "",
        "model": meta.get("gear_model") or "",
        "gear_type": meta.get("gear_type") or "",
        "tone_type": meta.get("tone_type") or "",
        "architecture": arch,
    }


def scan_folder(folder: str, use_cache: bool = True) -> tuple[list[dict], list[str]]:
    """Scan a folder (recursively) for .nam files. Returns (entries, errors)."""
    folder = os.path.abspath(folder)
    cache = _load_json(SCAN_CACHE_PATH, {}) if use_cache else {}
    entries, errors = [], []
    dirty = False
    for root, _dirs, files in os.walk(folder):
        for fn in sorted(files):
            if not fn.lower().endswith(".nam"):
                continue
            path = os.path.join(root, fn)
            try:
                mtime = os.path.getmtime(path)
                key = path
                cached = cache.get(key)
                if cached and cached.get("_mtime") == mtime:
                    entries.append({k: v for k, v in cached.items() if k != "_mtime"})
                    continue
                entry = _read_nam_entry(path)
                cache[key] = {**entry, "_mtime": mtime}
                dirty = True
                entries.append(entry)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{fn}: {type(e).__name__}: {e}")
    if dirty:
        try:
            _save_json(SCAN_CACHE_PATH, cache)
        except OSError:
            pass
    return entries, errors


def filter_entries(
    entries: list[dict],
    text: str = "",
    tone_type: str = "any",
    gear_type: str = "any",
) -> list[dict]:
    """Case-insensitive substring filter over name/make/model + type filters."""
    text = (text or "").strip().lower()
    out = []
    for e in entries:
        if text:
            hay = " ".join([e["name"], e["make"], e["model"], os.path.basename(e["path"])]).lower()
            if not all(t in hay for t in text.split()):
                continue
        if tone_type != "any" and e["tone_type"] != tone_type:
            continue
        if gear_type != "any" and e["gear_type"] != gear_type:
            continue
        out.append(e)
    return out


# ----------------------------------------------------------------------------
# Named libraries
# ----------------------------------------------------------------------------


def load_libraries() -> dict[str, list[str]]:
    return _load_json(LIBRARIES_PATH, {})


def save_library(name: str, paths: list[str]) -> None:
    name = name.strip()
    if not name:
        raise ValueError("Library name must not be empty.")
    if not paths:
        raise ValueError("Library must contain at least one .nam file.")
    libs = load_libraries()
    libs[name] = sorted(set(paths))
    _save_json(LIBRARIES_PATH, libs)


def delete_library(name: str) -> bool:
    libs = load_libraries()
    if name in libs:
        del libs[name]
        _save_json(LIBRARIES_PATH, libs)
        return True
    return False


def library_paths(name: str) -> list[str]:
    """Existing files in a named library (silently drops moved/deleted files)."""
    paths = load_libraries().get(name, [])
    return [p for p in paths if os.path.isfile(p)]
