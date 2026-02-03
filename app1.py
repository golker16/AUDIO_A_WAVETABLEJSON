#!/usr/bin/env python3
# WTGEN-1.2 exporter + reference interpreter (Python)
#
# Implements:
#  - Export: WAV -> WTGEN JSON with spectralData (harm-noise-framepack-v1)
#  - Load/Validate + DAG execution (graph)
#  - spectralData decode + reconstruction via IFFT (minimum phase)
#  - Ops: tilt, spectralMask, formantBank
#  - MacroParam.kind: affine, remap, choose, jitter
#  - perFrame (JSON Pointer -> ModStack): lfo, noise, drift, jitter
#  - perFramePack: sampleLike expander + deterministic band picking
#  - post: dcRemove + normalize
#
# Notes:
#  - Determinism is achieved via: xxhash32 + PCG32 + quantized spectral payload.
#  - This is a "reference engine" to prove the format. Your plugin can mirror this.

import argparse
import base64
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

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

# --- NEW: Evitar transitorios al inicio (C) ---
PICK_MIN_CENTER_MS = 200  # evita ataques/transitorios al inicio

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

# ----------------------------
# WTGEN-1.2 "spectralData" (Ruta 1)
# ----------------------------
DEFAULT_HARMONICS = 512   # magnitudes armónicas por frame (1..H)
DEFAULT_NOISE_BANDS = 48  # bandas para residual/ruido (coarse)
PHASE_MODE = "minimumPhase"  # declarativo: el plugin reconstruye fase

# cuantización normada
HARM_QUANT = "u16_q12"        # amp_q = round(amp * 4096)  (amp lineal)
NOISE_DB_QUANT = "i16_q0.5db" # db_q = round(db * 2)       (0.5 dB)

# límite de dB para residual
NOISE_DB_MIN = -64.0
NOISE_DB_MAX = +16.0


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


# ============================================================
# Determinism: xxhash32 + PCG32
# ============================================================

def _rotl32(x: int, r: int) -> int:
    return ((x << r) | (x >> (32 - r))) & 0xFFFFFFFF


def xxhash32(data: bytes, seed: int = 0) -> int:
    """
    Pure-python xxHash32 reference-ish implementation (sufficient for determinism).
    Matches typical xxhash32 behavior for byte streams.
    """
    # constants
    P1 = 0x9E3779B1
    P2 = 0x85EBCA77
    P3 = 0xC2B2AE3D
    P4 = 0x27D4EB2F
    P5 = 0x165667B1

    n = len(data)
    i = 0

    def round_acc(acc: int, inp: int) -> int:
        acc = (acc + (inp * P2 & 0xFFFFFFFF)) & 0xFFFFFFFF
        acc = _rotl32(acc, 13)
        acc = (acc * P1) & 0xFFFFFFFF
        return acc

    if n >= 16:
        v1 = (seed + P1 + P2) & 0xFFFFFFFF
        v2 = (seed + P2) & 0xFFFFFFFF
        v3 = (seed + 0) & 0xFFFFFFFF
        v4 = (seed - P1) & 0xFFFFFFFF

        limit = n - 16
        while i <= limit:
            d1 = struct.unpack_from("<I", data, i)[0]; i += 4
            d2 = struct.unpack_from("<I", data, i)[0]; i += 4
            d3 = struct.unpack_from("<I", data, i)[0]; i += 4
            d4 = struct.unpack_from("<I", data, i)[0]; i += 4
            v1 = round_acc(v1, d1)
            v2 = round_acc(v2, d2)
            v3 = round_acc(v3, d3)
            v4 = round_acc(v4, d4)

        h32 = (_rotl32(v1, 1) + _rotl32(v2, 7) + _rotl32(v3, 12) + _rotl32(v4, 18)) & 0xFFFFFFFF

        def merge_round(h: int, v: int) -> int:
            v = round_acc(0, v)
            h ^= v
            h = ((h * P1) + P4) & 0xFFFFFFFF
            return h

        h32 = merge_round(h32, v1)
        h32 = merge_round(h32, v2)
        h32 = merge_round(h32, v3)
        h32 = merge_round(h32, v4)
    else:
        h32 = (seed + P5) & 0xFFFFFFFF

    h32 = (h32 + n) & 0xFFFFFFFF

    # remaining 4-byte chunks
    while i + 4 <= n:
        k1 = struct.unpack_from("<I", data, i)[0]
        i += 4
        k1 = (k1 * P3) & 0xFFFFFFFF
        k1 = _rotl32(k1, 17)
        k1 = (k1 * P4) & 0xFFFFFFFF
        h32 ^= k1
        h32 = (_rotl32(h32, 17) * P1 + P4) & 0xFFFFFFFF

    # remaining bytes
    while i < n:
        k1 = data[i]
        i += 1
        h32 ^= (k1 * P5) & 0xFFFFFFFF
        h32 = (_rotl32(h32, 11) * P1) & 0xFFFFFFFF

    # avalanche
    h32 ^= (h32 >> 15)
    h32 = (h32 * P2) & 0xFFFFFFFF
    h32 ^= (h32 >> 13)
    h32 = (h32 * P3) & 0xFFFFFFFF
    h32 ^= (h32 >> 16)

    return h32 & 0xFFFFFFFF


class PCG32:
    """
    PCG32 (XSH RR) deterministic PRNG.
    """
    def __init__(self, seed: int = 0, inc: int = 54):
        self.state = 0
        self.inc = ((inc << 1) | 1) & 0xFFFFFFFFFFFFFFFF
        self.seed(seed)

    def seed(self, seed: int):
        self.state = 0
        self.random_u32()
        self.state = (self.state + (seed & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
        self.random_u32()

    def random_u32(self) -> int:
        oldstate = self.state
        self.state = (oldstate * 6364136223846793005 + self.inc) & 0xFFFFFFFFFFFFFFFF
        xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) & 0xFFFFFFFF
        rot = (oldstate >> 59) & 31
        return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF

    def random_float(self) -> float:
        return self.random_u32() / 4294967296.0  # [0,1)

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.random_float()

    def normalish(self) -> float:
        # quick approx gaussian via sum of uniforms (CLT)
        s = 0.0
        for _ in range(6):
            s += self.random_float()
        return (s - 3.0) / 1.0  # roughly ~N(0,1)


def seed_sub(root_seed: int, parts: List[Any]) -> int:
    """
    HASH32(rootSeed, parts...) with stable serialization (utf-8 + length prefix).
    """
    b = bytearray()
    b += struct.pack("<I", root_seed & 0xFFFFFFFF)
    for p in parts:
        if p is None:
            s = "null"
        elif isinstance(p, (int, np.integer)):
            s = f"i:{int(p)}"
        elif isinstance(p, (float, np.floating)):
            # stable float repr (avoid platform quirks): use hex float
            s = "f:" + float(p).hex()
        else:
            s = "s:" + str(p)
        bs = s.encode("utf-8")
        b += struct.pack("<I", len(bs))
        b += bs
    return xxhash32(bytes(b), seed=0)


# ============================================================
# Utility (exporter) helpers
# ============================================================

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
    w = win
    num = np.mean((x * w) ** 2)
    den = np.mean(w ** 2)
    return float(np.sqrt(num / max(den, 1e-12)))


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def parabolic_refine(y: np.ndarray, i: int) -> float:
    if i <= 0 or i >= len(y) - 1:
        return 0.0
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = (y0 - 2.0 * y1 + y2)
    if abs(denom) < 1e-12:
        return 0.0
    delta = 0.5 * (y0 - y2) / denom
    return float(np.clip(delta, -0.5, 0.5))


# ============================================================
# YIN (CMND) - robust period estimate (exporter)
# ============================================================

def yin_cmnd(x: np.ndarray, lag_min: int, lag_max: int) -> tuple[np.ndarray, np.ndarray]:
    x = x.astype(np.float64, copy=False)
    n = len(x)
    lag_max = min(lag_max, n - 2)
    lag_min = max(1, lag_min)
    if lag_max <= lag_min:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    taus = np.arange(lag_min, lag_max + 1)
    d = np.zeros_like(taus, dtype=np.float64)

    for i, tau in enumerate(taus):
        a = x[:n - tau]
        b = x[tau:]
        diff = a - b
        d[i] = np.dot(diff, diff)

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
    lag_min = int(math.floor(sr / f0_max))
    lag_max = int(math.ceil(sr / f0_min))
    lag_min = max(lag_min, int(sr * (YIN_MIN_PERIOD_MS / 1000.0)))

    _, cmnd = yin_cmnd(x, lag_min, lag_max)
    if len(cmnd) == 0:
        T = sr / 440.0
        return float(T), 440.0, 1.0

    best_idx = None
    for i in range(1, len(cmnd) - 1):
        if cmnd[i] < threshold and cmnd[i] <= cmnd[i - 1] and cmnd[i] <= cmnd[i + 1]:
            best_idx = i
            break

    if best_idx is None:
        best_idx = int(np.argmin(cmnd[1:-1])) + 1

    inv = -cmnd
    delta = parabolic_refine(inv, best_idx)
    lag = (lag_min + best_idx) + delta

    f0 = sr / lag if lag > 1e-6 else 440.0
    cmnd_min = float(np.min(cmnd))
    return float(lag), float(f0), cmnd_min


# ============================================================
# Peak pick + scoring (exporter)
# ============================================================

def find_peak_rms_candidates(x: np.ndarray, sr: int,
                             window_ms: float = PICK_WINDOW_MS,
                             hop_ms: float = PICK_HOP_MS,
                             topk: int = TOPK_CANDIDATES) -> list[tuple[int, float]]:
    win_len = max(16, int(round(sr * window_ms / 1000.0)))
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    win = hann_window(win_len)

    # --- NEW: offset de inicio para evitar picks en transitorios (C) ---
    min_center = int(round(sr * (PICK_MIN_CENTER_MS / 1000.0)))
    start0 = 0

    # Queremos que el CENTRO de la ventana sea >= min_center
    if len(x) > win_len and min_center < len(x):
        start0 = clamp_int(min_center - (win_len // 2), 0, max(0, len(x) - win_len))

    centers = []
    rms_vals = []
    crest_vals = []

    for start in range(start0, max(1, len(x) - win_len), hop):
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

    ok = crest_vals <= CREST_MAX

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

    idxs_sorted = idxs[np.argsort(rms_vals[idxs])[::-1]]
    idxs_sorted = idxs_sorted[:topk]

    return [(int(centers[i]), float(rms_vals[i])) for i in idxs_sorted]


def region_score(mono: np.ndarray, sr: int, center_idx: int) -> tuple[float, dict]:
    # --- NEW: clamp real del span (B.1) ---
    span = int(round(sr * (PICK_SPAN_MS / 1000.0)))
    span = max(1, min(span, len(mono)))  # clamp al tamaño real
    half = span // 2

    lo = clamp_int(center_idx - half, 0, len(mono) - 1)
    hi = clamp_int(center_idx + half, 0, len(mono) - 1)
    seg = mono[lo:hi].astype(np.float32)
    if len(seg) < int(sr * 0.1):
        seg = mono.astype(np.float32)
        lo = 0

    rms_local = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))

    T, f0, cmnd_min = estimate_period_yin(seg, sr, F0_MIN_HZ, F0_MAX_HZ, YIN_THRESHOLD)
    harmonicity = float(np.clip(1.0 - cmnd_min, 0.0, 1.0))

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


# ============================================================
# Phase anchor + extraction helpers (exporter)
# ============================================================

def find_rising_zero_crossing_near(x: np.ndarray, idx: int, search_radius: int) -> int:
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
    t = start + (np.arange(out_len, dtype=np.float64) * (period / out_len))
    t = np.clip(t, 0.0, n - 1.001)
    return cs(t).astype(np.float32)


def fine_align_start_spline(cs: CubicSpline, n: int, start: float, period: float, ref_cycle: np.ndarray, search: int) -> float:
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


# ============================================================
# spectralData exporter: harmonic + noise bands
# ============================================================

def _linear_band_edges(lo_bin: int, hi_bin: int, bands: int) -> list[tuple[int, int]]:
    if bands <= 0 or hi_bin < lo_bin:
        return []
    length = hi_bin - lo_bin + 1
    edges = []
    for b in range(bands):
        a = lo_bin + (b * length) // bands
        c = lo_bin + ((b + 1) * length) // bands - 1
        c = max(a, c)
        edges.append((a, c))
    return edges


def frames_to_harm_noise(frames: np.ndarray, harm_count: int, noise_bands: int) -> tuple[np.ndarray, np.ndarray, dict]:
    F, N = frames.shape
    max_harm = max(1, (N // 2) - 1)
    H = int(min(harm_count, max_harm))

    amp_scale = 2.0 / float(N)
    lo_bin = H + 1
    hi_bin = N // 2
    band_edges = _linear_band_edges(lo_bin, hi_bin, int(noise_bands))

    harm_q = np.zeros((F, H), dtype=np.uint16)
    noise_q = np.zeros((F, len(band_edges)), dtype=np.int16)

    eps = 1e-12

    for i in range(F):
        x = frames[i].astype(np.float64, copy=False)
        X = np.fft.rfft(x)
        mag = np.abs(X)

        harm_amp = mag[1:H + 1] * amp_scale
        harm_amp = np.clip(harm_amp, 0.0, 4.0)

        qh = np.round(harm_amp * 4096.0).astype(np.int64)
        qh = np.clip(qh, 0, 65535).astype(np.uint16)
        harm_q[i, :] = qh

        for b, (a, c) in enumerate(band_edges):
            seg = mag[a:c + 1] * amp_scale
            rms = float(np.sqrt(np.mean(seg * seg))) if seg.size else 0.0
            db = 20.0 * math.log10(max(rms, eps))
            db = float(np.clip(db, NOISE_DB_MIN, NOISE_DB_MAX))
            qdb = int(np.round(db * 2.0))  # 0.5 dB
            qdb = int(np.clip(qdb, -32768, 32767))
            noise_q[i, b] = np.int16(qdb)

    info = {
        "harmonicsCount": H,
        "noiseBands": int(len(band_edges)),
        "ampScale": "rfft_mag*(2/N)",
        "harmQuant": HARM_QUANT,
        "noiseDbQuant": NOISE_DB_QUANT,
        "noiseDbRange": [NOISE_DB_MIN, NOISE_DB_MAX],
        "banding": {
            "type": "linear_bins",
            "loBin": int(lo_bin),
            "hiBin": int(hi_bin),
        },
    }
    return harm_q, noise_q, info


def pack_harm_noise_framepack(harm_q: np.ndarray, noise_q: np.ndarray, table_size: int) -> str:
    F, H = harm_q.shape
    B = noise_q.shape[1]

    out = bytearray()
    out += b"HNFPv1\0"
    out += struct.pack("<HHHH", int(table_size), int(F), int(H), int(B))

    t_amount = 0
    t_center = 0
    t_width = 0

    for i in range(F):
        out += harm_q[i].astype("<u2", copy=False).tobytes(order="C")
        out += noise_q[i].astype("<i2", copy=False).tobytes(order="C")
        out += struct.pack("<HHH", t_amount, t_center, t_width)

    return base64.b64encode(bytes(out)).decode("ascii")


# ============================================================
# Exporter: WAV -> frames -> spectralData JSON
# ============================================================

def wav_to_frames_and_meta(wav_path: Path, sr_target: int, frames_n: int, table_size: int) -> tuple[np.ndarray, ImportMeta]:
    audio, sr0 = sf.read(str(wav_path), always_2d=True, dtype="float32")
    src_channels = audio.shape[1]
    mono = np.mean(audio, axis=1).astype(np.float32)
    duration_s = len(mono) / float(sr0)

    sr = int(sr0)
    if sr != sr_target:
        g = math.gcd(sr, sr_target)
        up = sr_target // g
        down = sr // g
        mono = resample_poly(mono, up=up, down=down).astype(np.float32)
        sr = int(sr_target)

    candidates = find_peak_rms_candidates(mono, sr, PICK_WINDOW_MS, PICK_HOP_MS, TOPK_CANDIDATES)

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

    span = int(round(sr * (PICK_SPAN_MS / 1000.0)))
    half = span // 2
    seg_lo = clamp_int(peak_idx - half, 0, len(mono) - 1)
    seg_hi = clamp_int(peak_idx + half, 0, len(mono) - 1)
    segment = mono[seg_lo:seg_hi].astype(np.float32)
    if len(segment) < int(sr * 0.1):
        segment = mono.astype(np.float32)
        seg_lo = 0

    T, f0, cmnd_min = estimate_period_yin(segment, sr, F0_MIN_HZ, F0_MAX_HZ, YIN_THRESHOLD)

    search_radius = int(max(16, round(2.0 * T)))
    local_peak = clamp_int(peak_idx - seg_lo, 0, len(segment) - 1)
    anchor_local = find_rising_zero_crossing_near(segment, local_peak, search_radius)
    anchor = seg_lo + anchor_local

    mid = (frames_n - 1) / 2.0
    starts = np.array([anchor + (k - mid) * T for k in range(frames_n)], dtype=np.float64)

    need_lo = int(np.floor(np.min(starts) - 2))
    need_hi = int(np.ceil(np.max(starts) + T + 2))
    need_lo = clamp_int(need_lo, 0, len(mono) - 1)
    need_hi = clamp_int(need_hi, 0, len(mono) - 1)
    work = mono[need_lo:need_hi].astype(np.float32)
    work_offset = need_lo

    xi = np.arange(len(work), dtype=np.float64)
    cs = CubicSpline(xi, work.astype(np.float64, copy=False), bc_type="natural")

    ref_start = float(starts[int(round(mid))] - work_offset)
    ref_cycle = extract_cycle_frac_spline(cs, len(work), ref_start, T, table_size)

    out_frames = np.zeros((frames_n, table_size), dtype=np.float32)
    for k in range(frames_n):
        s = float(starts[k] - work_offset)
        s2 = fine_align_start_spline(cs, len(work), s, T, ref_cycle, search=3)
        out_frames[k] = extract_cycle_frac_spline(cs, len(work), s2, T, table_size)

    out_frames = frame_smooth(out_frames, FRAME_SMOOTH)

    for k in range(frames_n):
        out_frames[k] = dc_remove_frame(out_frames[k], strength=1.0)

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


def build_wtgen_json_spectral(frames: np.ndarray,
                              meta: ImportMeta,
                              engine_name: str,
                              engine_version: str,
                              preset_name: str,
                              seed: int,
                              table_size: int,
                              frames_n: int,
                              harm_count: int,
                              noise_bands: int) -> dict:
    harm_q, noise_q, spec_info = frames_to_harm_noise(frames, harm_count=harm_count, noise_bands=noise_bands)
    b64 = pack_harm_noise_framepack(harm_q, noise_q, table_size=table_size)

    determinism = {
        "hash32": "xxhash32",
        "prng": "pcg32",
        "floatDeterminism": "quantized_payload",
        "macroParam": {"requireKind": True},
        "paramPath": "json_pointer",
        "postPolicy": {
            "postAlwaysLast": True,
            "opsDisallowed": ["normalize", "dcRemove"]
        }
    }

    macros = {"complexity": 0.5, "brightness": 0.5, "motion": 0.0}

    # --- NEW: spanMs efectivo (B.2) ---
    span_ms_effective = int(min(PICK_SPAN_MS, round(meta.duration_s * 1000.0)))

    doc = {
        "schema": "wtgen-1",
        "engine": {
            "name": engine_name,
            "version": engine_version,
            "determinism": determinism,
            "opVersions": {
                "spectralData": 1,
                "tilt": 1,
                "spectralMask": 1,
                "formantBank": 1
            }
        },
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
                "spanMs": span_ms_effective,  # NEW: metadata honesta
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
            }
        },

        "program": {
            "mode": "graph",
            "macros": macros,
            "nodes": [
                {
                    "id": "src",
                    "op": "spectralData",
                    "p": {
                        "codec": "harm-noise-framepack-v1",
                        "tableSize": int(table_size),
                        "frames": int(frames_n),
                        "channels": 1,
                        "harmonics": {
                            "count": int(spec_info["harmonicsCount"]),
                            "quant": spec_info["harmQuant"],
                            "ampScale": spec_info["ampScale"]
                        },
                        "noise": {
                            "bands": int(spec_info["noiseBands"]),
                            "quantDb": spec_info["noiseDbQuant"],
                            "dbRange": spec_info["noiseDbRange"],
                            "banding": spec_info["banding"]
                        },
                        "phase": {"mode": PHASE_MODE},
                        "data": b64
                    },
                    "perFrame": {},
                    "perFramePack": None
                }
            ],
            "out": "src"
        },

        "morph": {"frameInterp": "cubic", "phasePolicy": "unwrap_lock"},

        # --- NEW: quitar dcRemove/normalize del post (A) ---
        "post": {
            "loopPolish": {"mode": "minclick", "strength": 0.5}
        },

        "meta": {"name": preset_name, "tags": ["spectralData", "reconstructible", "deterministic"]}
    }
    return doc

