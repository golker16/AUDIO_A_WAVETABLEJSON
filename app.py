import argparse
import base64
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from scipy.interpolate import CubicSpline


# ----------------------------
# Config (v1 defaults)
# ----------------------------
DEFAULT_TABLE_SIZE = 2048
DEFAULT_FRAMES = 64
DEFAULT_SR = 48000

# "momento de mayor volumen"
PICK_WINDOW_MS = 50
PICK_HOP_MS = 10
PICK_SPAN_MS = 1500

# f0 range para ciclo
F0_MIN_HZ = 40.0
F0_MAX_HZ = 2000.0

# smoothing leve entre frames (opcional, determinista)
FRAME_SMOOTH = 0.15


@dataclass
class ImportMeta:
    sha256: str
    src_sr: int
    src_channels: int
    duration_s: float
    picked_sample: int
    picked_time_s: float
    f0_hz: float
    period_samples: float
    anchor_sample: int


# ----------------------------
# Utility
# ----------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hann_window(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones((n,), dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float64) / (n - 1))


def rms_hann(x: np.ndarray, win: np.ndarray) -> float:
    # win debe ser Hann. RMS = sqrt(mean((x*win)^2) / mean(win^2)) para compensar energía de ventana
    w = win
    num = np.mean((x * w) ** 2)
    den = np.mean(w ** 2)
    return float(np.sqrt(num / max(den, 1e-12)))


def find_peak_rms_index(x: np.ndarray, sr: int,
                        window_ms: float = PICK_WINDOW_MS,
                        hop_ms: float = PICK_HOP_MS) -> int:
    win_len = max(16, int(round(sr * window_ms / 1000.0)))
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    win = hann_window(win_len)

    best_i = 0
    best_val = -1.0

    # barrido por frames de análisis
    for start in range(0, max(1, len(x) - win_len), hop):
        seg = x[start:start + win_len]
        if len(seg) < win_len:
            break
        v = rms_hann(seg, win)
        if v > best_val:
            best_val = v
            best_i = start + win_len // 2  # centro de ventana
    return best_i


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def normalized_acf(x: np.ndarray, lag_min: int, lag_max: int) -> np.ndarray:
    """
    ACF normalizada por lag:
      r(l) = sum x[n]*x[n+l] / sqrt(sum x[n]^2 * sum x[n+l]^2)
    Computación simple (v1). Para spans pequeños va bien.
    """
    x = x.astype(np.float64, copy=False)
    n = len(x)
    lag_max = min(lag_max, n - 2)
    lag_min = max(1, lag_min)
    if lag_max <= lag_min:
        return np.zeros((0,), dtype=np.float64)

    acf = np.zeros((lag_max - lag_min + 1,), dtype=np.float64)
    for i, lag in enumerate(range(lag_min, lag_max + 1)):
        a = x[:n - lag]
        b = x[lag:]
        num = np.dot(a, b)
        den = math.sqrt(max(np.dot(a, a) * np.dot(b, b), 1e-18))
        acf[i] = num / den
    return acf


def parabolic_refine(y: np.ndarray, i: int) -> float:
    """
    Refina el máximo en índice i usando 3 puntos: i-1, i, i+1.
    Devuelve desplazamiento fraccional delta en [-0.5, 0.5] aprox.
    """
    if i <= 0 or i >= len(y) - 1:
        return 0.0
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = (y0 - 2.0 * y1 + y2)
    if abs(denom) < 1e-12:
        return 0.0
    delta = 0.5 * (y0 - y2) / denom
    return float(np.clip(delta, -0.5, 0.5))


def estimate_period_acf(x: np.ndarray, sr: int,
                        f0_min: float = F0_MIN_HZ,
                        f0_max: float = F0_MAX_HZ) -> tuple[float, float]:
    lag_min = int(math.floor(sr / f0_max))
    lag_max = int(math.ceil(sr / f0_min))
    acf = normalized_acf(x, lag_min, lag_max)
    if len(acf) == 0:
        # fallback: 440 Hz aproximado
        T = sr / 440.0
        return T, 440.0

    k = int(np.argmax(acf))
    delta = parabolic_refine(acf, k)
    lag = (lag_min + k) + delta
    f0 = sr / lag if lag > 1e-6 else 440.0
    return float(lag), float(f0)


def find_rising_zero_crossing_near(x: np.ndarray, idx: int, search_radius: int) -> int:
    """
    Busca cruce por cero ascendente (x[n] <= 0 y x[n+1] > 0) cerca de idx.
    """
    n = len(x)
    lo = clamp_int(idx - search_radius, 0, n - 2)
    hi = clamp_int(idx + search_radius, 0, n - 2)

    best = None
    best_dist = 10**18
    for i in range(lo, hi):
        if x[i] <= 0.0 and x[i + 1] > 0.0:
            d = abs(i - idx)
            if d < best_dist:
                best_dist = d
                best = i
    if best is None:
        # fallback: mínimo absoluto cercano
        seg = x[lo:hi+1]
        best = lo + int(np.argmin(np.abs(seg)))
    return int(best)


def extract_cycle_frac(x: np.ndarray, start: float, period: float, out_len: int) -> np.ndarray:
    """
    Extrae un ciclo de longitud 'period' desde 'start' (ambos float en samples),
    y lo re-muestrea a out_len usando CubicSpline.
    """
    n = len(x)
    # muestreo de puntos en el ciclo
    t = start + (np.arange(out_len, dtype=np.float64) * (period / out_len))
    # clamp de seguridad
    t = np.clip(t, 0.0, n - 1.001)

    # spline sobre señal completa (v1) — para performance podrías splinear por segmento
    # Para n grande, esto puede ser pesado; por eso primero recortamos alrededor del análisis.
    xi = np.arange(n, dtype=np.float64)
    cs = CubicSpline(xi, x.astype(np.float64, copy=False), bc_type="natural")
    y = cs(t).astype(np.float32)
    return y


def fine_align_start(x: np.ndarray, start: float, period: float, ref_cycle: np.ndarray, search: int) -> float:
    """
    Ajusta start dentro de +/-search samples para maximizar correlación con ref_cycle.
    """
    best_s = start
    best_c = -1e18
    # probamos offsets enteros alrededor
    for off in range(-search, search + 1):
        y = extract_cycle_frac(x, start + off, period, len(ref_cycle))
        c = float(np.dot(y, ref_cycle))
        if c > best_c:
            best_c = c
            best_s = start + off
    return float(best_s)


def dc_remove_frame(frame: np.ndarray, strength: float = 1.0) -> np.ndarray:
    m = float(np.mean(frame))
    return (frame - strength * m).astype(np.float32)


def normalize_global(frames: np.ndarray, target: float = 0.999) -> np.ndarray:
    peak = float(np.max(np.abs(frames)))
    if peak < 1e-12:
        return frames
    g = target / peak
    return (frames * g).astype(np.float32)


def frame_smooth(frames: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.0:
        return frames
    out = frames.copy()
    F = frames.shape[0]
    for i in range(F):
        a = frames[i]
        b = frames[i - 1] if i - 1 >= 0 else frames[i]
        c = frames[i + 1] if i + 1 < F else frames[i]
        out[i] = ((1.0 - amount) * a + 0.5 * amount * (b + c)).astype(np.float32)
    return out


def pack_q15_framepack(frames: np.ndarray) -> str:
    """
    frames: shape (F, N) float32 in [-1,1]
    q15: int16 little-endian, base64
    """
    x = np.clip(frames, -1.0, 1.0)
    q = np.round(x * 32767.0).astype(np.int16)  # Q15
    raw = q.tobytes(order="C")  # little-endian in numpy int16 on x86/win
    return base64.b64encode(raw).decode("ascii")


# ----------------------------
# Main pipeline: WAV -> WT frames
# ----------------------------
def wav_to_wavetable_frames(wav_path: Path,
                            sr_target: int,
                            frames_n: int,
                            table_size: int) -> tuple[np.ndarray, ImportMeta]:
    # load
    audio, sr = sf.read(str(wav_path), always_2d=True, dtype="float32")
    src_channels = audio.shape[1]
    mono = np.mean(audio, axis=1)  # L+R/2 si stereo, o mismo si mono

    duration_s = len(mono) / float(sr)

    # resample to target for consistency
    if sr != sr_target:
        # resample_poly with fixed rational approximation
        # up/down determined by gcd
        g = math.gcd(sr, sr_target)
        up = sr_target // g
        down = sr // g
        mono = resample_poly(mono, up=up, down=down).astype(np.float32)
        sr = sr_target

    # pick peak loudness moment
    peak_idx = find_peak_rms_index(mono, sr, PICK_WINDOW_MS, PICK_HOP_MS)

    # analysis segment
    span = int(round(sr * (PICK_SPAN_MS / 1000.0)))
    half = span // 2
    seg_lo = clamp_int(peak_idx - half, 0, len(mono) - 1)
    seg_hi = clamp_int(peak_idx + half, 0, len(mono) - 1)
    segment = mono[seg_lo:seg_hi].astype(np.float32)
    if len(segment) < int(sr * 0.1):
        # muy corto: usa todo
        segment = mono.astype(np.float32)
        seg_lo = 0

    # estimate period
    T, f0 = estimate_period_acf(segment, sr, F0_MIN_HZ, F0_MAX_HZ)

    # phase anchor near peak within segment: rising zero crossing
    # search radius ~ 2 periods
    search_radius = int(max(16, round(2.0 * T)))
    local_peak = clamp_int(peak_idx - seg_lo, 0, len(segment) - 1)
    anchor_local = find_rising_zero_crossing_near(segment, local_peak, search_radius)
    anchor = seg_lo + anchor_local

    # build starts for cycles centered around anchor
    mid = (frames_n - 1) / 2.0
    starts = np.array([anchor + (k - mid) * T for k in range(frames_n)], dtype=np.float64)

    # we will spline on a working window around the needed range for performance
    # Determine range needed to cover all starts + T
    need_lo = int(np.floor(np.min(starts) - 2))
    need_hi = int(np.ceil(np.max(starts) + T + 2))
    need_lo = clamp_int(need_lo, 0, len(mono) - 1)
    need_hi = clamp_int(need_hi, 0, len(mono) - 1)
    work = mono[need_lo:need_hi].astype(np.float32)
    work_offset = need_lo

    # reference cycle = central
    ref_start = float(starts[int(round(mid))] - work_offset)
    ref_cycle = extract_cycle_frac(work, ref_start, T, table_size)

    # extract all cycles with fine alignment
    out_frames = np.zeros((frames_n, table_size), dtype=np.float32)
    for k in range(frames_n):
        s = float(starts[k] - work_offset)
        # align within +/- 3 samples (en sr 48k es muy fino)
        s2 = fine_align_start(work, s, T, ref_cycle, search=3)
        out_frames[k] = extract_cycle_frac(work, s2, T, table_size)

    # optional smoothing across frames
    out_frames = frame_smooth(out_frames, FRAME_SMOOTH)

    # DC remove per frame (recommended)
    for k in range(frames_n):
        out_frames[k] = dc_remove_frame(out_frames[k], strength=1.0)

    # normalize global to 0.999 (recommended)
    out_frames = normalize_global(out_frames, target=0.999)

    meta = ImportMeta(
        sha256=sha256_file(wav_path),
        src_sr=int(sr_target if sr_target else sr),
        src_channels=int(src_channels),
        duration_s=float(duration_s),
        picked_sample=int(peak_idx),
        picked_time_s=float(peak_idx / sr),
        f0_hz=float(f0),
        period_samples=float(T),
        anchor_sample=int(anchor),
    )
    return out_frames, meta


def build_wtgen_json(frames: np.ndarray,
                     meta: ImportMeta,
                     engine_name: str,
                     engine_version: str,
                     preset_name: str,
                     seed: int,
                     table_size: int,
                     frames_n: int) -> dict:
    b64 = pack_q15_framepack(frames)

    doc = {
        "schema": "wtgen-1",
        "engine": {"name": engine_name, "version": engine_version},
        "wt": {"tableSize": int(table_size), "frames": int(frames_n), "channels": 1},
        "seed": int(seed),

        "import": {
            "source": {
                "type": "wav",
                "sha256": meta.sha256,
                "channels": meta.src_channels,
                "sr": meta.src_sr,
                "durationSec": meta.duration_s
            },
            "pre": {"downmix": "L+R/2", "resampleTo": DEFAULT_SR},
            "pick": {"metric": "rms_hann", "windowMs": PICK_WINDOW_MS, "hopMs": PICK_HOP_MS, "spanMs": PICK_SPAN_MS,
                     "pickedTimeSec": meta.picked_time_s},
            "cycle": {"method": "acf_norm_parabolic", "f0MinHz": F0_MIN_HZ, "f0MaxHz": F0_MAX_HZ,
                      "anchor": "rising_zc", "f0Hz": meta.f0_hz, "periodSamples": meta.period_samples},
            "build": {"cyclePolicy": "consecutive_centered", "resample": "cubic", "frameSmooth": FRAME_SMOOTH,
                      "anchorSample": meta.anchor_sample}
        },

        "program": {
            "mode": "graph",
            "nodes": [
                {
                    "id": "src",
                    "op": "tableData",
                    "p": {
                        "codec": "q15-framepack-v1",
                        "tableSize": int(table_size),
                        "frames": int(frames_n),
                        "channels": 1,
                        "data": b64
                    }
                }
            ],
            "out": "src"
        },

        "morph": {"frameInterp": "cubic", "phasePolicy": "unwrap_lock"},

        "post": {
            "dcRemove": True,
            "normalize": {"mode": "peak", "target": 0.999, "scope": "global"},
            "loopPolish": {"mode": "minclick", "strength": 0.5}
        },

        "meta": {"name": preset_name, "tags": ["wav", "fidelity", "peakmatch"]}
    }
    return doc


def main():
    ap = argparse.ArgumentParser(description="WAV -> WTGEN-1 JSON wavetable (peak RMS pick, cycle extraction)")
    ap.add_argument("input_wav", type=str, help="Input .wav path")
    ap.add_argument("-o", "--output", type=str, default="", help="Output .json path")
    ap.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    ap.add_argument("--tableSize", type=int, default=DEFAULT_TABLE_SIZE)
    ap.add_argument("--sr", type=int, default=DEFAULT_SR)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--name", type=str, default="FromWavPeak")
    ap.add_argument("--engineName", type=str, default="wt_exe")
    ap.add_argument("--engineVersion", type=str, default="1.0.0")
    args = ap.parse_args()

    in_path = Path(args.input_wav).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    out_path = Path(args.output).expanduser().resolve() if args.output else in_path.with_suffix(".wtgen.json")

    frames, meta = wav_to_wavetable_frames(
        wav_path=in_path,
        sr_target=int(args.sr),
        frames_n=int(args.frames),
        table_size=int(args.tableSize)
    )

    doc = build_wtgen_json(
        frames=frames,
        meta=meta,
        engine_name=args.engineName,
        engine_version=args.engineVersion,
        preset_name=args.name,
        seed=args.seed,
        table_size=int(args.tableSize),
        frames_n=int(args.frames)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote: {out_path}")
    print(f"     peakTime={meta.picked_time_s:.3f}s  f0~{meta.f0_hz:.2f}Hz  period~{meta.period_samples:.2f} samples")


if __name__ == "__main__":
    main()
