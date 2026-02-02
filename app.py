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

# ----------------------------
# Mejoras v1.1 (cerradas)
# ----------------------------
TOPK_CANDIDATES = 8

# Transient guard
CREST_MAX = 8.0          # si peak/rms > esto, probable click/transitorio
PERSIST_NEIGHBOR = 2     # requiere RMS alto también en vecinos (suaviza picks espurios)

# YIN
YIN_THRESHOLD = 0.15     # umbral típico: 0.10-0.20
YIN_MIN_PERIOD_MS = 0.5  # evita periodos absurdos (muy alta freq)


@dataclass
class ImportMeta:
    sha256: str
    src_sr_original: int
    proc_sr: int
    src_channels: int
    duration_s: float
    picked_sample: int
    picked_time_s: float
    f0_hz: float
    period_samples: float
    anchor_sample: int
    cmnd_min: float
    pick_score: float


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
    # RMS = sqrt(mean((x*win)^2) / mean(win^2)) para compensar energía de ventana
    w = win
    num = np.mean((x * w) ** 2)
    den = np.mean(w ** 2)
    return float(np.sqrt(num / max(den, 1e-12)))


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


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


# ----------------------------
# YIN (CMND) - robust period estimate
# ----------------------------
def yin_cmnd(x: np.ndarray, lag_min: int, lag_max: int) -> tuple[np.ndarray, np.ndarray]:
    """
    YIN: Difference function d(tau) y CMND(tau)
    Retorna (d, cmnd) arrays para tau in [lag_min..lag_max]
    """
    x = x.astype(np.float64, copy=False)
    n = len(x)
    lag_max = min(lag_max, n - 2)
    lag_min = max(1, lag_min)
    if lag_max <= lag_min:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    taus = np.arange(lag_min, lag_max + 1)
    d = np.zeros_like(taus, dtype=np.float64)

    # d(tau) = sum (x[n] - x[n+tau])^2
    for i, tau in enumerate(taus):
        a = x[:n - tau]
        b = x[tau:]
        diff = a - b
        d[i] = np.dot(diff, diff)

    # CMND(tau) = d(tau) / ((1/tau) * sum_{j=1..tau} d(j))
    cmnd = np.zeros_like(d)
    running = 0.0
    for i in range(len(d)):
        running += d[i]
        denom = running / (i + 1)
        cmnd[i] = d[i] / max(denom, 1e-18)

    return d, cmnd


def estimate_period_yin(x: np.ndarray, sr: int,
                        f0_min: float = F0_MIN_HZ,
                        f0_max: float = F0_MAX_HZ,
                        threshold: float = YIN_THRESHOLD) -> tuple[float, float, float]:
    """
    Retorna (period_samples, f0_hz, cmnd_min)
    period_samples es float refinado.
    """
    lag_min = int(math.floor(sr / f0_max))
    lag_max = int(math.ceil(sr / f0_min))

    # evita lags demasiado chicos
    lag_min = max(lag_min, int(sr * (YIN_MIN_PERIOD_MS / 1000.0)))

    _, cmnd = yin_cmnd(x, lag_min, lag_max)
    if len(cmnd) == 0:
        T = sr / 440.0
        return float(T), 440.0, 1.0

    # buscar primer tau que baja del umbral y que sea un mínimo local
    best_idx = None
    for i in range(1, len(cmnd) - 1):
        if cmnd[i] < threshold and cmnd[i] <= cmnd[i - 1] and cmnd[i] <= cmnd[i + 1]:
            best_idx = i
            break

    # fallback: elegir mínimo absoluto de CMND
    if best_idx is None:
        best_idx = int(np.argmin(cmnd[1:-1])) + 1

    # refinamiento parabólico sobre cmnd (mínimo): invertimos signo y refinamos máximo
    inv = -cmnd
    delta = parabolic_refine(inv, best_idx)
    lag = (lag_min + best_idx) + delta

    f0 = sr / lag if lag > 1e-6 else 440.0
    cmnd_min = float(np.min(cmnd))
    return float(lag), float(f0), cmnd_min


# ----------------------------
# Peak pick: RMS Hann + transient guard + Top-K
# ----------------------------
def find_peak_rms_candidates(x: np.ndarray, sr: int,
                             window_ms: float = PICK_WINDOW_MS,
                             hop_ms: float = PICK_HOP_MS,
                             topk: int = TOPK_CANDIDATES) -> list[tuple[int, float]]:
    """
    Devuelve lista de candidatos (center_index, rms) ordenados por rms desc.
    Aplica transient guard (crest factor) y persistencia.
    """
    win_len = max(16, int(round(sr * window_ms / 1000.0)))
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    win = hann_window(win_len)

    centers = []
    rms_vals = []
    crest_vals = []

    for start in range(0, max(1, len(x) - win_len), hop):
        seg = x[start:start + win_len]
        if len(seg) < win_len:
            break

        r = rms_hann(seg, win)
        p = float(np.max(np.abs(seg)))
        crest = p / max(r, 1e-12)

        centers.append(start + win_len // 2)
        rms_vals.append(r)
        crest_vals.append(crest)

    if not rms_vals:
        return [(len(x) // 2, 0.0)]

    rms_vals = np.array(rms_vals, dtype=np.float64)
    crest_vals = np.array(crest_vals, dtype=np.float64)

    # transient guard
    ok = crest_vals <= CREST_MAX

    # persistencia: el candidato debe estar alto también en vecinos
    if PERSIST_NEIGHBOR > 0:
        ok2 = ok.copy()
        for i in range(len(ok)):
            if not ok[i]:
                continue
            lo = max(0, i - PERSIST_NEIGHBOR)
            hi = min(len(ok) - 1, i + PERSIST_NEIGHBOR)
            neigh = float(np.max(rms_vals[lo:hi + 1]))
            if neigh < 0.85 * float(rms_vals[i]):
                ok2[i] = False
        ok = ok2

    idxs = np.where(ok)[0]
    if len(idxs) == 0:
        idxs = np.arange(len(rms_vals))

    # top-k por RMS
    idxs_sorted = idxs[np.argsort(rms_vals[idxs])[::-1]]
    idxs_sorted = idxs_sorted[:topk]

    return [(int(centers[i]), float(rms_vals[i])) for i in idxs_sorted]


# ----------------------------
# Multi-region scoring
# ----------------------------
def region_score(mono: np.ndarray, sr: int, center_idx: int) -> tuple[float, dict]:
    """
    Score = RMS_local + harmonicity + stability
    Retorna (score, debug_dict)
    """
    span = int(round(sr * (PICK_SPAN_MS / 1000.0)))
    half = span // 2
    lo = clamp_int(center_idx - half, 0, len(mono) - 1)
    hi = clamp_int(center_idx + half, 0, len(mono) - 1)
    seg = mono[lo:hi].astype(np.float32)
    if len(seg) < int(sr * 0.1):
        seg = mono.astype(np.float32)
        lo = 0

    # RMS local (sin Hann; score relativo)
    rms_local = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))

    # YIN principal
    T, f0, cmnd_min = estimate_period_yin(seg, sr, F0_MIN_HZ, F0_MAX_HZ, YIN_THRESHOLD)
    harmonicity = float(np.clip(1.0 - cmnd_min, 0.0, 1.0))

    # estabilidad: medir T en 3 subventanas
    thirds = np.array_split(seg, 3)
    Ts = []
    for part in thirds:
        if len(part) < int(sr * 0.05):
            continue
        t_i, _, _ = estimate_period_yin(part, sr, F0_MIN_HZ, F0_MAX_HZ, YIN_THRESHOLD)
        Ts.append(t_i)

    if len(Ts) >= 2:
        Ts = np.array(Ts, dtype=np.float64)
        rel_var = float(np.std(Ts) / max(np.mean(Ts), 1e-9))
        stability = float(np.clip(1.0 - 3.0 * rel_var, 0.0, 1.0))
    else:
        stability = 0.0

    # score combinado (pesos fijos v1.1)
    score = (1.0 * rms_local) + (0.6 * harmonicity) + (0.4 * stability)

    dbg = {
        "rmsLocal": rms_local,
        "f0Hz": f0,
        "periodSamples": T,
        "cmndMin": cmnd_min,
        "harmonicity": harmonicity,
        "stability": stability
    }
    return score, dbg


# ----------------------------
# Phase anchor + extraction helpers
# ----------------------------
def find_rising_zero_crossing_near(x: np.ndarray, idx: int, search_radius: int) -> int:
    """
    Busca cruce por cero ascendente (x[n] <= 0 y x[n+1] > 0) cerca de idx.
    """
    n = len(x)
    lo = clamp_int(idx - search_radius, 0, n - 2)
    hi = clamp_int(idx + search_radius, 0, n - 2)

    best = None
    best_dist = 10 ** 18
    for i in range(lo, hi):
        if x[i] <= 0.0 and x[i + 1] > 0.0:
            d = abs(i - idx)
            if d < best_dist:
                best_dist = d
                best = i

    if best is None:
        seg = x[lo:hi + 1]
        best = lo + int(np.argmin(np.abs(seg)))
    return int(best)


def extract_cycle_frac_spline(cs: CubicSpline, n: int, start: float, period: float, out_len: int) -> np.ndarray:
    """
    Extrae un ciclo usando spline precomputado (performance).
    """
    t = start + (np.arange(out_len, dtype=np.float64) * (period / out_len))
    t = np.clip(t, 0.0, n - 1.001)
    return cs(t).astype(np.float32)


def fine_align_start_spline(cs: CubicSpline, n: int, start: float, period: float, ref_cycle: np.ndarray, search: int) -> float:
    """
    Ajusta start dentro de +/-search samples para maximizar correlación con ref_cycle.
    """
    best_s = start
    best_c = -1e18
    for off in range(-search, search + 1):
        y = extract_cycle_frac_spline(cs, n, start + off, period, len(ref_cycle))
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
    raw = q.tobytes(order="C")
    return base64.b64encode(raw).decode("ascii")


# ----------------------------
# Main pipeline: WAV -> WT frames
# ----------------------------
def wav_to_wavetable_frames(wav_path: Path,
                            sr_target: int,
                            frames_n: int,
                            table_size: int) -> tuple[np.ndarray, ImportMeta]:
    # load
    audio, sr0 = sf.read(str(wav_path), always_2d=True, dtype="float32")
    src_channels = audio.shape[1]
    mono = np.mean(audio, axis=1).astype(np.float32)  # L+R/2 si stereo

    duration_s = len(mono) / float(sr0)

    # resample to target for consistency
    sr = int(sr0)
    if sr != sr_target:
        g = math.gcd(sr, sr_target)
        up = sr_target // g
        down = sr // g
        mono = resample_poly(mono, up=up, down=down).astype(np.float32)
        sr = int(sr_target)

    # 1) candidatos de "más volumen" (RMS Hann) con transient guard
    candidates = find_peak_rms_candidates(mono, sr, PICK_WINDOW_MS, PICK_HOP_MS, TOPK_CANDIDATES)

    # 2) multi-region: elegir mejor candidato por harmonicidad/estabilidad
    best = None
    best_score = -1e18
    best_dbg = None

    for center_idx, _cand_rms in candidates:
        score, dbg = region_score(mono, sr, center_idx)
        if (score > best_score) or (abs(score - best_score) < 1e-12 and (best is None or center_idx < best)):
            best_score = score
            best = center_idx
            best_dbg = dbg

    peak_idx = int(best if best is not None else candidates[0][0])

    # analysis segment alrededor del peak_idx elegido
    span = int(round(sr * (PICK_SPAN_MS / 1000.0)))
    half = span // 2
    seg_lo = clamp_int(peak_idx - half, 0, len(mono) - 1)
    seg_hi = clamp_int(peak_idx + half, 0, len(mono) - 1)
    segment = mono[seg_lo:seg_hi].astype(np.float32)
    if len(segment) < int(sr * 0.1):
        segment = mono.astype(np.float32)
        seg_lo = 0

    # 3) periodo con YIN (más robusto)
    T, f0, cmnd_min = estimate_period_yin(segment, sr, F0_MIN_HZ, F0_MAX_HZ, YIN_THRESHOLD)

    # phase anchor near peak within segment: rising zero crossing
    search_radius = int(max(16, round(2.0 * T)))
    local_peak = clamp_int(peak_idx - seg_lo, 0, len(segment) - 1)
    anchor_local = find_rising_zero_crossing_near(segment, local_peak, search_radius)
    anchor = seg_lo + anchor_local

    # build starts for cycles centered around anchor
    mid = (frames_n - 1) / 2.0
    starts = np.array([anchor + (k - mid) * T for k in range(frames_n)], dtype=np.float64)

    # working window around needed range
    need_lo = int(np.floor(np.min(starts) - 2))
    need_hi = int(np.ceil(np.max(starts) + T + 2))
    need_lo = clamp_int(need_lo, 0, len(mono) - 1)
    need_hi = clamp_int(need_hi, 0, len(mono) - 1)
    work = mono[need_lo:need_hi].astype(np.float32)
    work_offset = need_lo

    # spline ONCE (performance)
    xi = np.arange(len(work), dtype=np.float64)
    cs = CubicSpline(xi, work.astype(np.float64, copy=False), bc_type="natural")

    # reference cycle = central
    ref_start = float(starts[int(round(mid))] - work_offset)
    ref_cycle = extract_cycle_frac_spline(cs, len(work), ref_start, T, table_size)

    # extract all cycles with fine alignment
    out_frames = np.zeros((frames_n, table_size), dtype=np.float32)
    for k in range(frames_n):
        s = float(starts[k] - work_offset)
        s2 = fine_align_start_spline(cs, len(work), s, T, ref_cycle, search=3)
        out_frames[k] = extract_cycle_frac_spline(cs, len(work), s2, T, table_size)

    # optional smoothing across frames
    out_frames = frame_smooth(out_frames, FRAME_SMOOTH)

    # DC remove per frame
    for k in range(frames_n):
        out_frames[k] = dc_remove_frame(out_frames[k], strength=1.0)

    # normalize global
    out_frames = normalize_global(out_frames, target=0.999)

    picked_time_s = float(peak_idx / sr) if sr > 0 else 0.0
    cmnd_meta = float(best_dbg["cmndMin"]) if best_dbg and "cmndMin" in best_dbg else float(cmnd_min)

    meta = ImportMeta(
        sha256=sha256_file(wav_path),
        src_sr_original=int(sr0),
        proc_sr=int(sr),
        src_channels=int(src_channels),
        duration_s=float(duration_s),
        picked_sample=int(peak_idx),
        picked_time_s=picked_time_s,
        f0_hz=float(f0),
        period_samples=float(T),
        anchor_sample=int(anchor),
        cmnd_min=cmnd_meta,
        pick_score=float(best_score),
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
                "srOriginal": meta.src_sr_original,
                "srProcessed": meta.proc_sr,
                "durationSec": meta.duration_s
            },
            "pre": {"downmix": "L+R/2", "resampleTo": DEFAULT_SR},
            "pick": {
                "metric": "rms_hann",
                "windowMs": PICK_WINDOW_MS,
                "hopMs": PICK_HOP_MS,
                "spanMs": PICK_SPAN_MS,
                "crestMax": CREST_MAX,
                "topK": TOPK_CANDIDATES,
                "pickedTimeSec": meta.picked_time_s,
                "pickScore": meta.pick_score
            },
            "cycle": {
                "method": "yin_cmnd",
                "f0MinHz": F0_MIN_HZ,
                "f0MaxHz": F0_MAX_HZ,
                "threshold": YIN_THRESHOLD,
                "anchor": "rising_zc",
                "f0Hz": meta.f0_hz,
                "periodSamples": meta.period_samples,
                "cmndMin": meta.cmnd_min
            },
            "build": {
                "cyclePolicy": "consecutive_centered",
                "resample": "cubic",
                "frameSmooth": FRAME_SMOOTH,
                "anchorSample": meta.anchor_sample
            }
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
    ap = argparse.ArgumentParser(description="WAV -> WTGEN-1 JSON wavetable (RMS Hann pick + YIN CMND + multi-region)")
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
    print(
        f"     pickedTime={meta.picked_time_s:.3f}s  "
        f"score={meta.pick_score:.3f}  "
        f"f0~{meta.f0_hz:.2f}Hz  "
        f"period~{meta.period_samples:.2f} samples  "
        f"cmndMin={meta.cmnd_min:.3f}"
    )


if __name__ == "__main__":
    main()

