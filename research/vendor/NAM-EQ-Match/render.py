"""Replay an exported preset on a new DI without per-file normalization."""
import argparse
from tonematch.pipeline import render_preset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--di", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--channel", choices=("left", "right", "mean"))
    args = parser.parse_args()
    render_preset(args.settings, args.di, args.out, device=args.device, channel=args.channel)
