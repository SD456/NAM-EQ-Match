"""Local-only constrained NAM retrieval CLI. Candidates require a manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True)
    ap.add_argument("--di", required=True)
    ap.add_argument("--manifest", required=True, help="JSON list: path, rig_type, optional cab_ir_path")
    ap.add_argument("--out", required=True, help="New or empty directory")
    ap.add_argument("--gain-range", type=float, nargs=2, default=(-12, 12))
    ap.add_argument("--refine-top", type=int, default=5)
    ap.add_argument("--render-top", type=int, default=1)
    ap.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    ap.add_argument("--target-channel", choices=("left", "right", "mean"))
    ap.add_argument("--di-channel", choices=("left", "right", "mean"))
    ap.add_argument("--target-start", type=float, default=0)
    ap.add_argument("--target-end", type=float)
    ap.add_argument("--preview-s", type=float, default=30)
    ap.add_argument("--di-normalization-db", type=float, help="Fixed calibration from a previous run")
    ap.add_argument("--eq-limit-db", type=float, default=4.5)
    args = ap.parse_args()
    from tonematch.nam_backend import NamCapture, resolve_device
    from tonematch.pipeline import run_match

    manifest_path = Path(args.manifest).resolve()
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, list) or not manifest:
        ap.error("Manifest must be a nonempty list")
    captures = []
    device = resolve_device(args.device)
    for item in manifest:
        if item.get("rig_type") not in ("amp-only", "full-rig"):
            ap.error("Each candidate requires verified rig_type: amp-only or full-rig")
        model = (manifest_path.parent / item["path"]).resolve()
        cab = item.get("cab_ir_path")
        cab = str((manifest_path.parent / cab).resolve()) if cab else None
        captures.append(NamCapture(str(model), device=device, cab_ir_path=cab, rig_type=item["rig_type"]))
    result = run_match(args.target, args.di, captures, args.out,
                       gain_range_db=tuple(args.gain_range), refine_top=args.refine_top,
                       render_top=args.render_top, preview_s=args.preview_s,
                       target_channel=args.target_channel, di_channel=args.di_channel,
                       target_start_s=args.target_start, target_end_s=args.target_end,
                       di_normalization_db=args.di_normalization_db, eq_max_gain_db=args.eq_limit_db,
                       progress_cb=lambda f, m: print(f"[{f:6.1%}] {m}"))
    print(f"Best refined candidate: {result.best.name}; score={result.best.score:.6f}")
    print(f"Report: {result.report_path}")
    print(result.report["how_to_use"])


if __name__ == "__main__":
    main()
