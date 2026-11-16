"""Freeze, then calibrate a clean single-note frontend with instrument-held-out labels."""
from __future__ import annotations

import argparse
import importlib.metadata
import inspect
from pathlib import Path
import platform
import sys
import time
import xml.etree.ElementTree as ET

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stage2_benchmark import read_json, sha, write_json
from scripts.frontend_calibration import estimate_frontend, detect_onsets, pitch_metrics, match_onsets


def validate_config(cfg):
    """This study supports one frozen profile; descriptive fields must be true."""
    fixed = {
        "pitch_windows":[1024,2048],"thresholds":[0,.1,.2,.4,.6,.8],
        "baseline":{"pitch_window":1024,"threshold":.8},
        "frontend":{"analysis_sr":16000,"hop":256,"fmin":82.4,"fmax":1319,"resolution":.5,
                    "center":True,"rms_window":1024,"relative_rms_floor":.01,
                    "rms_percentile":95,"shared_edge_seconds":.064},
        "truth":{"midi_range":[40,88],"a4_hz":440,"correct_cents":50,"guard_seconds":.114,
                 "offset_rounding_tolerance_seconds":.00005,"registers_midi":[[40,51],[52,63],[64,75],[76,88]],
                 "phases_seconds":{"attack":[0,.08],"sustain":[.12,.4]},"note_equal_weighting":True},
        "onset":{"primary_window":1024,"diagnostic_window":2048,"tolerances_seconds":[.05,.1],
                 "backtrack":False,"matching":"ordered maximum cardinality one-to-one on all full-file predictions"},
        "selection":{"min_coverage":.6,"min_conditional_exact":.9,
                     "order":["max full_correct","min conditional_octave","min pitch_window","max threshold"],
                     "no_feasible":"abstain"},
        "gate":{"min_coverage":.6,"min_conditional_exact":.9,"min_onset_f1":.8,"onset_tolerance_seconds":.05}}
    for key,value in fixed.items():
        if cfg[key] != value:
            raise ValueError(f"Unsupported stage6 profile in {key}; update protocol and implementation together")


def recording_pairs(manifest, cfg):
    """Read labels/headers only; waveform analysis begins after the freeze."""
    entries = manifest["selection"]
    if len(entries) != 312 or any("sha256" not in e or "output" not in e for e in entries):
        raise ValueError("Require 156 completely downloaded original WAV/XML pairs")
    assets, grouped = {}, {}
    for entry in entries:
        path = "assets/stage6/" + entry["output"]
        if sha(ROOT / path) != entry["sha256"]:
            raise ValueError(f"Changed downloaded source: {path}")
        assets[path] = entry["sha256"]
        key = (entry["split"], Path(path).stem)
        grouped.setdefault(key, {})[Path(path).suffix.lower()] = path
        if cfg["split"][entry["split"]] not in entry["name"]:
            raise ValueError("Frozen instrument split mismatch")
    records, wave_hashes = [], set()
    for (split, identifier), paths in sorted(grouped.items()):
        if set(paths) != {".wav", ".xml"}:
            raise ValueError("Unpaired WAV/XML")
        info = sf.info(ROOT / paths[".wav"])
        native = cfg["native_audio"]
        if (info.samplerate != native["sample_rate"] or info.channels != native["channels"]
                or info.subtype != native["subtype"]):
            raise ValueError("Require original native 44.1k mono PCM16")
        xml = ET.parse(ROOT / paths[".xml"]).getroot()
        events = xml.findall("./transcription/event")
        if (xml.findtext("./globalParameter/instrument") != "EGUI" or len(events) != 1
                or xml.findtext("./globalParameter/audioFileName") != Path(paths[".wav"]).name):
            raise ValueError("Require exactly one paired electric-guitar note")
        event = events[0]
        if event.findtext("excitationStyle") != "PK" or event.findtext("expressionStyle") != "NO":
            raise ValueError("Require ordinary picked notes without expression techniques")
        pitch, string, fret = (int(event.findtext(k)) for k in ("pitch", "stringNumber", "fretNumber"))
        onset, original_offset = (float(event.findtext(k)) for k in ("onsetSec", "offsetSec"))
        duration = info.frames / info.samplerate
        if (not 1 <= string <= 6 or not 0 <= fret <= 12
                or pitch != [40,45,50,55,59,64][string-1] + fret
                or not 40 <= pitch <= 88 or not 0 <= onset < original_offset
                or original_offset > duration + cfg["truth"]["offset_rounding_tolerance_seconds"]):
            raise ValueError("Invalid label pitch, string/fret or time bounds")
        digest = assets[paths[".wav"]]
        if digest in wave_hashes:
            raise ValueError("Duplicate audio cannot appear as independent records")
        wave_hashes.add(digest)
        records.append({"id": identifier, "split": split, "wav": paths[".wav"], "xml": paths[".xml"],
                        "sample_rate": info.samplerate, "frames": info.frames, "string": string, "fret": fret,
                        "original_offset": original_offset,
                        "event": {"pitch":pitch,"onset":onset,"offset":min(original_offset,duration)}})
    for split in cfg["split"]:
        group = [r for r in records if r["split"] == split]
        if (len(group) != cfg["expected_recordings_per_split"]
                or len({(r["string"],r["fret"]) for r in group}) != 78
                or len({r["event"]["pitch"] for r in group}) < 3):
            raise ValueError("Require all 78 string/fret notes for each instrument")
        guard = cfg["truth"]["guard_seconds"]
        stable_notes = 0
        for record in group:
            count = 1 + int(np.ceil(record["frames"]*16000/record["sample_rate"]))//256
            times = np.arange(count)*.016
            event = record["event"]
            stable_notes += bool(np.any((times >= event["onset"]+guard)
                & (times < event["offset"]-guard) & (times >= .064)
                & (times < record["frames"]/record["sample_rate"]-.064)))
        if stable_notes < 12:
            raise ValueError("Insufficient potentially scoreable notes")
    return records, assets


def plan(cfg):
    validate_config(cfg)
    protected = read_json(ROOT / cfg["baseline_snapshot"])
    for path,digest in protected.items():
        if sha(ROOT/path) != digest:
            raise ValueError(f"Protected baseline changed: {path}")
    records, assets = recording_pairs(read_json(ROOT/cfg["data_manifest"]), cfg)
    sources = ["experiments/stage6.json","docs/stage6-protocol.md","docs/stage6-data-alternatives.md",
               "scripts/stage6_benchmark.py","scripts/frontend_calibration.py","scripts/prepare_stage6_di.py",
               "tests/test_frontend_calibration.py","tests/test_stage6_benchmark.py",
               "scripts/stage2_benchmark.py",cfg["data_manifest"],cfg["baseline_snapshot"],
               "assets/stage6/source-description.pdf"]
    return {"config":cfg,"records":records,"assets":assets,"protected_baselines":protected,
            "source_hashes":{p:sha(ROOT/p) for p in sources},
            "environment":{"python":platform.python_version(),"platform":platform.platform(),
                "versions":{n:importlib.metadata.version(n) for n in ("numpy","scipy","librosa","soundfile")},
                "pyin_signature":str(inspect.signature(librosa.pyin)),
                "pyin_source_sha256":sha(inspect.getsourcefile(librosa.pyin)),
                "onset_source_sha256":sha(inspect.getsourcefile(librosa.onset.onset_detect))}}


def combine_pitch(rows):
    keys = ("n_notes","n_notes_excluded_empty_core","total_reference_frames","accepted_weight",
            "correct_weight","octave_weight","n_notes_with_any_accept","n_notes_with_correct")
    sums = {key:sum(r[key] for r in rows) for key in keys}
    n,a,c,o = (sums[k] for k in ("n_notes","accepted_weight","correct_weight","octave_weight"))
    sums.update(coverage=a/n if n else None,conditional_exact=c/a if a else None,
                full_correct=c/n if n else None,conditional_octave=o/a if a else None)
    bins = []
    for i in range(6):
        buckets = [r["confidence_bins"][i] for r in rows]
        weight = sum(b["weight"] for b in buckets)
        correct = sum(b["correct_weight"] for b in buckets)
        bins.append({"lower":[0,.1,.2,.4,.6,.8][i],"upper":[.1,.2,.4,.6,.8,1.00001][i],
                     "n_frames":sum(b["n_frames"] for b in buckets),"weight":weight,
                     "correct_weight":correct,"conditional_exact":correct/weight if weight else None})
    return {**sums,"confidence_bins":bins}


def choose_setting(grid, cfg):
    feasible = [x for x in grid if x["metrics"]["coverage"] is not None
                and x["metrics"]["coverage"] >= cfg["min_coverage"]
                and x["metrics"]["conditional_exact"] is not None
                and x["metrics"]["conditional_exact"] >= cfg["min_conditional_exact"]]
    if not feasible:
        return None
    best = min(feasible,key=lambda x:(-x["metrics"]["full_correct"],x["metrics"]["conditional_octave"],
                                      x["pitch_window"],-x["threshold"]))
    return {"pitch_window":best["pitch_window"],"threshold":best["threshold"]}


def combine_onsets(rows):
    tp,fp,fn = (sum(r[k] for r in rows) for k in ("tp","fp","fn"))
    errors = [e for r in rows for e in r["errors_seconds"]]
    return {"tp":tp,"fp":fp,"fn":fn,"precision":tp/(tp+fp) if tp+fp else 0,
            "recall":tp/(tp+fn) if tp+fn else 0,"f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
            "median_signed_ms":float(np.median(errors)*1000) if errors else None,
            "median_absolute_ms":float(np.median(np.abs(errors))*1000) if errors else None,
            "same_note_repeated_event_count":0}


def evaluate(records,cfg):
    rows = []
    for i,record in enumerate(records):
        audio,sr = sf.read(ROOT/record["wav"],dtype="float64")
        if audio.ndim != 1 or sr != record["sample_rate"] or len(audio) != record["frames"] or not np.isfinite(audio).all():
            raise ValueError("Audio differs from locked mono metadata")
        silent = not np.any(audio)
        onsets = {str(w):[] if silent else detect_onsets(audio,sr,w).tolist() for w in (1024,2048)}
        estimates = {}
        for window in cfg["pitch_windows"]:
            if silent:
                times = np.arange(1+int(np.ceil(len(audio)*16000/sr))//256)*.016
                estimates[window] = {"times":times,"f0":np.full(len(times),np.nan),
                    "raw_voiced":np.zeros(len(times),bool),"probability":np.zeros(len(times)),
                    "rms_ok":np.zeros(len(times),bool),"edge_ok":(times>=.064)&(times<len(audio)/sr-.064),
                    "duration_seconds":len(audio)/sr}
            else:
                estimates[window] = estimate_frontend(audio,sr,window)
        metrics = []
        for window,estimate in estimates.items():
            for threshold in cfg["thresholds"]:
                phases = {}
                for phase,(start,end) in cfg["truth"]["phases_seconds"].items():
                    event = record["event"]
                    sub = {"pitch":event["pitch"],"onset":event["onset"]+start,
                           "offset":min(event["offset"],event["onset"]+end)}
                    phases[phase] = pitch_metrics(estimate,[sub],threshold,guard_seconds=0)
                metrics.append({"pitch_window":window,"threshold":threshold,
                    "metrics":pitch_metrics(estimate,[record["event"]],threshold,cfg["truth"]["guard_seconds"]),
                    "phases":phases})
        onset_metrics = [{"window":w,"tolerance":tol,
                          "metrics":match_onsets(onsets[str(w)],[record["event"]["onset"]],tol)}
                         for w in (1024,2048) for tol in cfg["onset"]["tolerances_seconds"]]
        rows.append({"id":record["id"],"event":record["event"],"silent":silent,
                     "pitch":metrics,"predicted_onsets":onsets,"onsets":onset_metrics,
                     "raw_voiced_frames":{str(w):int(e["raw_voiced"].sum()) for w,e in estimates.items()}})
        if (i+1)%13 == 0:
            print(f'{record["split"]} frontend {i+1}/{len(records)}',flush=True)
    grid = []
    for window in cfg["pitch_windows"]:
        for threshold in cfg["thresholds"]:
            group = [next(x for x in row["pitch"] if x["pitch_window"]==window and x["threshold"]==threshold) for row in rows]
            registers = {f"{lo}-{hi}":combine_pitch([x["metrics"] for row,x in zip(rows,group)
                                                  if lo<=row["event"]["pitch"]<=hi])
                         for lo,hi in cfg["truth"]["registers_midi"]}
            grid.append({"pitch_window":window,"threshold":threshold,
                         "metrics":combine_pitch([x["metrics"] for x in group]),"registers":registers,
                         "phases":{p:combine_pitch([x["phases"][p] for x in group]) for p in ("attack","sustain")}})
    onsets = [{"window":w,"tolerance":tol,"metrics":combine_onsets([
        next(x["metrics"] for x in row["onsets"] if x["window"]==w and x["tolerance"]==tol) for row in rows])}
        for w in (1024,2048) for tol in cfg["onset"]["tolerances_seconds"]]
    return {"rows":rows,"grid":grid,"onsets":onsets}


def final_gate(heldout, selected, cfg):
    item = next((x for x in heldout["grid"] if selected is not None
                 and x["pitch_window"]==selected["pitch_window"] and x["threshold"]==selected["threshold"]),None)
    onset = next(x["metrics"] for x in heldout["onsets"] if x["window"]==1024 and x["tolerance"]==.05)
    metrics = item["metrics"] if item else {}
    checks = {"selection_available":selected is not None,
              "coverage":metrics.get("coverage") is not None and metrics["coverage"]>=cfg["min_coverage"],
              "conditional_exact":metrics.get("conditional_exact") is not None and metrics["conditional_exact"]>=cfg["min_conditional_exact"],
              "short_onset_f1":onset["f1"]>=cfg["min_onset_f1"]}
    return {"checks":checks,"passed":all(checks.values()),
            "decision":"eligible_for_separate_wet_frontend_validation" if all(checks.values()) else "retain_baseline",
            "production_replacement":False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze",action="store_true")
    args = parser.parse_args()
    cfg = read_json(ROOT/"experiments/stage6.json")
    output = ROOT/cfg["output"]
    if (output/"results.json").exists():
        raise FileExistsError("Completed calibration exists; refusing overwrite")
    current = plan(cfg)
    lock = output/"locked-plan.json"
    if lock.exists() and read_json(lock)!=current:
        raise ValueError("Frozen plan mismatch; do not adapt this run to observed labels")
    if args.freeze:
        output.mkdir(parents=True,exist_ok=True)
        write_json(lock,current)
        print("Frozen stage6; only labels, headers and bytes inspected",flush=True)
        return
    if not lock.exists():
        raise FileNotFoundError("Run --freeze before analyzing real DI")
    identity = sha(lock)
    started = time.perf_counter()
    calibration_path = output/"calibration.json"
    if calibration_path.exists():
        calibration = read_json(calibration_path)
        if calibration["lock_sha256"]!=identity:
            raise ValueError("Calibration cache lock mismatch")
    else:
        calibration = {"lock_sha256":identity,**evaluate([r for r in current["records"] if r["split"]=="calibration"],cfg)}
        write_json(calibration_path,calibration)
    selected = choose_setting(calibration["grid"],cfg["selection"])
    selection = {"lock_sha256":identity,"calibration_sha256":sha(calibration_path),"selected":selected,
                 "criterion":cfg["selection"],"heldout_used":False}
    selection_path = output/"selected-setting.json"
    if selection_path.exists() and read_json(selection_path)!=selection:
        raise ValueError("Persisted calibration selection changed")
    write_json(selection_path,selection)
    selection_identity = sha(selection_path)
    print(f"Locked calibration choice before heldout: {selected}",flush=True)
    heldout = evaluate([r for r in current["records"] if r["split"]=="heldout"],cfg)
    if plan(cfg)!=current or sha(lock)!=identity or sha(selection_path)!=selection_identity:
        raise ValueError("Frozen sources changed during calibration")
    write_json(output/"results.json",{"plan":current,"lock_sha256":identity,
        "selection":selection,"selection_sha256":selection_identity,"calibration":calibration,
        "heldout":heldout,"gate":final_gate(heldout,selected,cfg["gate"]),
        "baseline_preserved":True,"seconds":time.perf_counter()-started,
        "scope":"clean mono DI; heldout instrument, not heldout player or wet tone; no Free Bird"})
    print(f'Completed stage6: {final_gate(heldout,selected,cfg["gate"])}',flush=True)


if __name__ == "__main__":
    main()
