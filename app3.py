#!/usr/bin/env python3
# CLI wrapper extracted from app_full.py
#
# Keeps the command-line interface thin; heavy logic is in app1.py (export) and app2.py (render).

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from app1 import (
    DEFAULT_TABLE_SIZE,
    DEFAULT_FRAMES,
    DEFAULT_SR,
    DEFAULT_HARMONICS,
    DEFAULT_NOISE_BANDS,
    wav_to_frames_and_meta,
    build_wtgen_json_spectral,
)
from app2 import run_graph


def cmd_export(args):
    in_path = Path(args.input_wav).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    out_path = Path(args.output).expanduser().resolve() if args.output else in_path.with_suffix(".wtgen.json")

    frames, meta = wav_to_frames_and_meta(
        wav_path=in_path,
        sr_target=int(args.sr),
        frames_n=int(args.frames),
        table_size=int(args.tableSize),
    )

    doc = build_wtgen_json_spectral(
        frames=frames,
        meta=meta,
        engine_name=args.engineName,
        engine_version=args.engineVersion,
        preset_name=args.name,
        seed=args.seed,
        table_size=int(args.tableSize),
        frames_n=int(args.frames),
        harm_count=int(args.harmonics),
        noise_bands=int(args.noiseBands),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(f"[OK] Exported WTGEN JSON -> {out_path}")


def cmd_render(args):
    json_path = Path(args.input_json).expanduser().resolve()
    if not json_path.exists():
        raise SystemExit(f"Input not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        doc = json.load(f)

    frames = run_graph(doc)  # (F,N)
    wt = doc["wt"]
    N = int(wt["tableSize"])
    F = int(wt["frames"])

    out_wav = Path(args.output).expanduser().resolve() if args.output else json_path.with_suffix(".render.wav")

    # Write as concatenated frames (F*N samples) at a nominal SR (configurable)
    sr = int(args.sr)
    audio = frames.reshape((F * N,)).astype(np.float32)
    sf.write(str(out_wav), audio, sr)

    print(f"[OK] Rendered wavetable: frames={F} tableSize={N} -> {out_wav}")

    if args.dump_frames:
        out_npy = out_wav.with_suffix(".frames.npy")
        np.save(str(out_npy), frames)
        print(f"[OK] Saved frames array: {out_npy}")


def build_argparser():
    ap = argparse.ArgumentParser(description="WTGEN exporter + reference interpreter (spectralData + ops + perFrame)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # export
    ex = sub.add_parser("export", help="WAV -> WTGEN JSON (spectralData)")
    ex.add_argument("input_wav", type=str, help="Input .wav path (solo para análisis/export)")
    ex.add_argument("-o", "--output", type=str, default="", help="Output .json path")
    ex.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    ex.add_argument("--tableSize", type=int, default=DEFAULT_TABLE_SIZE)
    ex.add_argument("--sr", type=int, default=DEFAULT_SR)
    ex.add_argument("--seed", type=int, default=1)
    ex.add_argument("--name", type=str, default="FromWavPeak_SpectralData")
    ex.add_argument("--engineName", type=str, default="wt_exe")
    ex.add_argument("--engineVersion", type=str, default="1.2.0")
    ex.add_argument("--harmonics", type=int, default=DEFAULT_HARMONICS)
    ex.add_argument("--noiseBands", type=int, default=DEFAULT_NOISE_BANDS)
    ex.set_defaults(func=cmd_export)

    # render
    rn = sub.add_parser("render", help="WTGEN JSON -> WAV (reference engine)")
    rn.add_argument("input_json", type=str, help="Input .wtgen.json path")
    rn.add_argument("-o", "--output", type=str, default="", help="Output .wav path (concatenated frames)")
    rn.add_argument("--sr", type=int, default=DEFAULT_SR, help="Output sample rate for rendered WAV")
    rn.add_argument("--dump-frames", action="store_true", help="Also save frames to .npy next to wav")
    rn.set_defaults(func=cmd_render)

    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
