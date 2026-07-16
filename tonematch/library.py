"""Library backend — scan .nam files, classify metadata, save/load named libraries.

Libraries are stored as directories under ``libraries/`` containing symlinks (or
hardlinks / copies) to the original .nam files.  No JSON manifest needed — the
directory itself IS the library.
"""

from __future__ import annotations

import json
import os
import shutil
import time

LIBRARIES_DIR = os.path.abspath("libraries")

BRAND_PATTERNS: dict[str, list[str]] = {
    "Mesa Boogie": ["mesa", "boogie", "rectifier", "mark ", "lonestar"],
    "Marshall": ["marshall", "jcm", "jvm", "jmp", "plexi"],
    "Fender": ["fender", "deluxe", "twin", "bassman", "princeton"],
    "Vox": ["vox", "ac30", "ac15"],
    "Orange": ["orange", "or15", "rockerverb"],
    "Friedman": ["friedman", "be-100"],
    "Diezel": ["diezel"],
    "ENGL": ["engl", "fireball"],
    "Bogner": ["bogner", "ecstasy"],
    "Peavey": ["peavey", "peavy", "6505", "5150"],
    "EVH": ["evh"],
    "Soldano": ["soldano", "slo"],
    "Hiwatt": ["hiwatt"],
    "Darkglass": ["darkglass"],
    "Ampeg": ["ampeg", "svt"],
    "Matchless": ["matchless"],
    "Dumble": ["dumble"],
    "Fortin": ["fortin"],
    "Blackstar": ["blackstar"],
    "Laney": ["laney"],
    "Victory": ["victory"],
    "Revv": ["revv"],
    "Morgan": ["morgan"],
    "Synergy": ["synergy"],
}

TONE_KEYWORDS: dict[str, list[str]] = {
    "Metal": ["metal", "metalcore", "djent", "death", "thrash", "chug", "doom", "sludge"],
    "High-Gain": ["high-gain", "high gain", "drive", "boost", "hot", "saturated", "mean", "lead", "distortion"],
    "Crunch": ["crunch", "edge", "breakup", "break up", "overdrive"],
    "Clean": ["clean", "acoustic", "pristine", "clear", "ambient", "worship"],
    "Fuzz": ["fuzz"],
}

BASS_KEYWORDS = {"ampeg", "darkglass", "hartke", "aguilar", "markbass", "gallien", "bass"}
BAD_SENTINELS = {"t3k-unset", "t3k-null", ""}


def _match_keywords(text: str, collection: dict[str, list[str]]) -> str | None:
    t = text.lower()
    for key, keywords in collection.items():
        for kw in keywords:
            if kw in t:
                return key
    return None


def parse_nam(filepath: str) -> dict | None:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("metadata", {})
        name = meta.get("name", "").strip()
        return {
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "display_name": name if name and name.lower() not in BAD_SENTINELS else os.path.splitext(os.path.basename(filepath))[0],
            "raw_gear_make": meta.get("gear_make", ""),
            "raw_gear_model": meta.get("gear_model", ""),
            "raw_gear_type": meta.get("gear_type", ""),
            "raw_tone_type": meta.get("tone_type", ""),
        }
    except Exception:
        return None


def resolve_brand(info: dict) -> str:
    make = info["raw_gear_make"].strip().lower()
    if make and make not in BAD_SENTINELS:
        result = _match_keywords(make, BRAND_PATTERNS)
        if result:
            return result
        return make.title() if make else "Unknown"
    result = _match_keywords(info["filename"], BRAND_PATTERNS)
    return result or "Unknown"


def resolve_instrument(info: dict) -> str:
    tone = info["raw_tone_type"].lower()
    if "bass" in tone:
        return "Bass"
    make = info["raw_gear_make"].strip().lower()
    if make and any(kw in make for kw in BASS_KEYWORDS):
        return "Bass"
    if "bass" in info["filename"].lower():
        return "Bass"
    return "Guitar"


def normalize_tone(info: dict) -> str:
    tone = info.get("raw_tone_type", "").strip().lower()
    if not tone or tone in BAD_SENTINELS:
        return "Unknown"
    result = _match_keywords(tone, TONE_KEYWORDS)
    return result or "Other"


def classify(info: dict) -> dict:
    info["brand"] = resolve_brand(info)
    info["tone_category"] = normalize_tone(info)
    info["instrument"] = resolve_instrument(info)
    info["type"] = info.get("raw_gear_type", "").strip() or "Unknown"
    return info


def scan_folder(path: str) -> list[dict]:
    excluded = os.path.normpath(os.path.join(path, "0 Export Selected"))
    items = []
    for root, _dirs, files in os.walk(path):
        if os.path.normpath(root).startswith(excluded):
            continue
        for f in files:
            if f.lower().endswith(".nam"):
                fp = os.path.join(root, f)
                info = parse_nam(fp)
                if info is not None:
                    items.append(classify(info))
    items.sort(key=lambda x: x["display_name"].lower())
    return items


# ---------------------------------------------------------------------------
# Library persistence — symlink-based
# ---------------------------------------------------------------------------

def _ensure_dir():
    os.makedirs(LIBRARIES_DIR, exist_ok=True)


def _link_file(src: str, dst: str) -> None:
    """Create a symlink, falling back to hardlink then copy."""
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError):
        try:
            os.link(src, dst)
        except (OSError, NotImplementedError):
            shutil.copy2(src, dst)


def save_library(name: str, file_paths: list[str]) -> str:
    _ensure_dir()
    sanitized = name.strip().replace("/", "_").replace("\\", "_").replace(".", "_")
    if not sanitized:
        raise ValueError("Library name cannot be empty.")
    target = os.path.join(LIBRARIES_DIR, sanitized)
    os.makedirs(target, exist_ok=True)
    for fp in file_paths:
        if os.path.isfile(fp):
            basename = os.path.basename(fp)
            dest = os.path.join(target, basename)
            if not os.path.exists(dest):
                _link_file(fp, dest)
    return sanitized


def load_library_path(name: str) -> str:
    """Return the folder path for a saved library."""
    return os.path.join(LIBRARIES_DIR, name)


def load_library_file_paths(name: str) -> list[str]:
    """Return all .nam file paths found in the library directory."""
    path = load_library_path(name)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Library '{name}' not found.")
    result = []
    for root, _dirs, files in os.walk(path):
        for f in files:
            if f.lower().endswith(".nam"):
                result.append(os.path.join(root, f))
    return sorted(result)


def list_libraries() -> list[str]:
    _ensure_dir()
    return sorted(
        e for e in os.listdir(LIBRARIES_DIR)
        if os.path.isdir(os.path.join(LIBRARIES_DIR, e)) and not e.startswith(".")
    )


def delete_library(name: str) -> None:
    path = os.path.join(LIBRARIES_DIR, name)
    if os.path.isdir(path):
        shutil.rmtree(path)


def library_file_count(name: str) -> int:
    path = os.path.join(LIBRARIES_DIR, name)
    if not os.path.isdir(path):
        return 0
    count = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            if f.lower().endswith(".nam"):
                count += 1
    return count


# ---------------------------------------------------------------------------
# Helpers for the Gradio Dataframe
# ---------------------------------------------------------------------------

def scan_for_dataframe(items: list[dict]) -> list[list]:
    return [
        [
            i["display_name"],
            i["brand"],
            i["tone_category"],
            i["instrument"],
            i["type"],
            i["filename"],
        ]
        for i in items
    ]
