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
    span = int(round(sr * (PICK_SPAN_MS / 1000.0)))
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

        "post": {
            "dcRemove": True,
            "normalize": {"mode": "peak", "target": 0.999, "scope": "global"},
            "loopPolish": {"mode": "minclick", "strength": 0.5}
        },

        "meta": {"name": preset_name, "tags": ["spectralData", "reconstructible", "deterministic"]}
    }
    return doc


# ============================================================
# Reference Engine: JSON -> wavetable frames (graph)
# ============================================================

class WTGenError(Exception):
    pass


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def smoothstep(u: float) -> float:
    u = clamp01(u)
    return u * u * (3.0 - 2.0 * u)


def curve_eval(name: str, u: float, param: float = 6.0) -> float:
    u = clamp01(u)
    if name == "linear":
        return u
    if name == "smooth":
        return smoothstep(u)
    if name == "exp2":
        k = float(param)
        num = (2.0 ** (k * u) - 1.0)
        den = (2.0 ** k - 1.0)
        return float(num / max(den, 1e-12))
    if name == "pow":
        p = float(param)
        return float(u ** p)
    return u


# ----------------------------
# JSON Pointer get/set
# ----------------------------

def _unescape_json_pointer_token(tok: str) -> str:
    return tok.replace("~1", "/").replace("~0", "~")


def parse_json_pointer(ptr: str) -> List[Any]:
    if ptr == "" or ptr == "/":
        return []
    if not ptr.startswith("/"):
        raise WTGenError(f"paramPath must be JSON Pointer starting with '/': {ptr}")
    parts = ptr.split("/")[1:]
    out: List[Any] = []
    for p in parts:
        p = _unescape_json_pointer_token(p)
        # array index?
        if p.isdigit():
            out.append(int(p))
        else:
            out.append(p)
    return out


def json_get(root: Any, ptr: str) -> Any:
    parts = parse_json_pointer(ptr)
    cur = root
    for t in parts:
        if isinstance(t, int):
            cur = cur[t]
        else:
            cur = cur[t]
    return cur


def json_set(root: Any, ptr: str, value: Any) -> None:
    parts = parse_json_pointer(ptr)
    if not parts:
        raise WTGenError("Cannot set root with empty pointer")
    cur = root
    for t in parts[:-1]:
        cur = cur[t] if isinstance(t, int) else cur[t]
    last = parts[-1]
    if isinstance(last, int):
        cur[last] = value
    else:
        cur[last] = value


# ----------------------------
# MacroParam evaluator
# ----------------------------

def is_macro_param(v: Any) -> bool:
    return isinstance(v, dict) and "kind" in v


def _apply_quantize(x: Any, q: Optional[dict]) -> Any:
    if q is None:
        return x
    mode = q.get("mode", "none")
    if mode == "none":
        return x
    if mode == "int":
        return int(round(float(x)))
    if mode == "step":
        step = float(q.get("step", 1.0))
        if step <= 0:
            return x
        return float(round(float(x) / step) * step)
    return x


def eval_macro_param(v: Any,
                     macros: Dict[str, float],
                     root_seed: int,
                     node_id: str,
                     param_path: str,
                     op_version: int,
                     t: float) -> Any:
    """
    Evaluate a literal or MacroParam. Supports: affine, remap, choose, jitter.
    For jitter: produces time-varying value via smooth noise as function of t.
    """
    if not is_macro_param(v):
        return v

    kind = v.get("kind")
    clamp_v = v.get("clamp", None)
    quant = v.get("quantize", None)

    def clamp_local(x: float) -> float:
        if clamp_v is None:
            return float(x)
        lo, hi = float(clamp_v[0]), float(clamp_v[1])
        return float(max(lo, min(hi, x)))

    if kind == "affine":
        base = float(v.get("base", 0.0))
        terms = v.get("terms", None)
        if terms is None:
            by = v.get("by", None)
            if by is None:
                # compatibility: "scaleBy" shape
                by = v.get("scaleBy", None)
            if by is None:
                val = base
            else:
                mname = by.get("macro", "complexity")
                amt = float(by.get("amount", 0.0))
                m = clamp01(float(macros.get(mname, 0.0)))
                val = base + amt * m
        else:
            s = 0.0
            for term in terms:
                mname = term.get("macro", "complexity")
                amt = float(term.get("amount", 0.0))
                m = clamp01(float(macros.get(mname, 0.0)))
                s += amt * m
            val = base + s

        val = clamp_local(val)
        val = _apply_quantize(val, quant)
        return val

    if kind == "remap":
        mname = v.get("macro", "complexity")
        x = clamp01(float(macros.get(mname, 0.0)))
        in0, in1 = v.get("in", [0.0, 1.0])
        out0, out1 = v.get("out", [0.0, 1.0])
        in0, in1 = float(in0), float(in1)
        out0, out1 = float(out0), float(out1)
        if abs(in1 - in0) < 1e-12:
            u = 0.0
        else:
            u = (x - in0) / (in1 - in0)
        u = clamp01(u)
        curve = v.get("curve", "linear")
        u2 = curve_eval(curve, u, param=float(v.get("k", 6.0)))
        val = out0 + (out1 - out0) * u2
        val = clamp_local(val)
        val = _apply_quantize(val, quant)
        return val

    if kind == "choose":
        by = v.get("by", "complexity")
        m = clamp01(float(macros.get(by, 0.0)))
        choices = v.get("choices", [])
        if not choices:
            return 0
        bias = float(v.get("bias", 0.0))
        interp = v.get("interp", "none")
        m = clamp01(m + bias)
        pos = m * (len(choices) - 1)
        if interp == "linear" and len(choices) >= 2:
            i0 = int(math.floor(pos))
            i1 = min(i0 + 1, len(choices) - 1)
            f = pos - i0
            a = float(choices[i0])
            b = float(choices[i1])
            val = a + (b - a) * f
        else:
            idx = int(math.floor(pos + 1e-12))
            idx = max(0, min(len(choices) - 1, idx))
            val = float(choices[idx])

        val = clamp_local(val)
        val = _apply_quantize(val, quant)
        return val

    if kind == "jitter":
        base = float(v.get("base", 0.0))
        depth_spec = v.get("depth", {"macro": "motion", "amount": 0.1})
        if isinstance(depth_spec, dict) and "macro" in depth_spec:
            m = clamp01(float(macros.get(depth_spec.get("macro", "motion"), 0.0)))
            depth = float(depth_spec.get("amount", 0.0)) * m
        else:
            depth = float(depth_spec)

        rate_spec = v.get("rate", {"macro": "complexity", "out": [0.5, 6.0]})
        if isinstance(rate_spec, dict) and "macro" in rate_spec:
            mname = rate_spec.get("macro", "complexity")
            m = clamp01(float(macros.get(mname, 0.0)))
            out0, out1 = rate_spec.get("out", [0.5, 6.0])
            rate = float(out0) + (float(out1) - float(out0)) * curve_eval(rate_spec.get("curve", "linear"), m)
        else:
            rate = float(rate_spec)

        noise_type = v.get("type", "smooth")
        seed_key = v.get("seedKey", "macroJitter")

        s = seed_sub(root_seed, [node_id, param_path, "macroParam", kind, seed_key, op_version])
        pr = PCG32(seed=s)

        # value noise on [0..1] with "rate" segments
        # We'll compute a stable smooth noise.
        def noise_value(tt: float) -> float:
            if rate <= 1e-9:
                # constant
                pr2 = PCG32(seed=s)
                return pr2.uniform(-1.0, 1.0)
            x = clamp01(tt) * rate
            i0 = int(math.floor(x))
            frac = x - i0
            # sample deterministic at i0 and i0+1 using derived seeds
            s0 = seed_sub(s, [i0, "n0"])
            s1 = seed_sub(s, [i0 + 1, "n1"])
            r0 = PCG32(seed=s0).uniform(-1.0, 1.0)
            r1 = PCG32(seed=s1).uniform(-1.0, 1.0)
            if noise_type == "step":
                return r0
            u = smoothstep(frac)
            return r0 + (r1 - r0) * u

        n = noise_value(t)
        val = base + depth * n
        val = clamp_local(val)
        val = _apply_quantize(val, quant)
        return val

    raise WTGenError(f"Unsupported MacroParam.kind: {kind}")


# ----------------------------
# perFrame ModStack
# ----------------------------

def eval_lfo(shape: str, rate: float, depth: float, phase: float, center: float, t: float) -> float:
    x = (t * rate + phase) % 1.0
    if shape == "sine":
        w = math.sin(2.0 * math.pi * x)
    elif shape == "tri":
        w = 4.0 * abs(x - 0.5) - 1.0
    elif shape == "saw":
        w = 2.0 * x - 1.0
    elif shape == "square":
        w = 1.0 if x < 0.5 else -1.0
    else:
        w = math.sin(2.0 * math.pi * x)
    return center + depth * w


def eval_value_noise(root_seed: int, node_id: str, param_path: str, op_version: int,
                     mod_type: str, seed_key: str, t: float,
                     rate: float, depth: float, mode: str) -> float:
    # sub-seed
    s = seed_sub(root_seed, [node_id, param_path, "perFrame", mod_type, seed_key, op_version])

    if rate <= 1e-12:
        r = PCG32(seed=s).uniform(-1.0, 1.0)
        return depth * r

    x = clamp01(t) * rate
    i0 = int(math.floor(x))
    frac = x - i0

    s0 = seed_sub(s, [i0, "a"])
    s1 = seed_sub(s, [i0 + 1, "b"])
    r0 = PCG32(seed=s0).uniform(-1.0, 1.0)
    r1 = PCG32(seed=s1).uniform(-1.0, 1.0)

    if mode == "step":
        return depth * r0

    u = smoothstep(frac)
    v = r0 + (r1 - r0) * u
    return depth * v


def eval_modulator(mod: dict,
                   macros: Dict[str, float],
                   root_seed: int,
                   node_id: str,
                   param_path: str,
                   op_version: int,
                   t: float,
                   frame_dt: float) -> float:
    """
    Returns delta contribution (not including mode add/mul/replace).
    """
    if "lfo" in mod:
        o = mod["lfo"]
        shape = str(o.get("shape", "sine"))
        rate = float(o.get("rate", 0.1))
        depth = float(o.get("depth", 0.0))
        phase = float(o.get("phase", 0.0))
        center = float(o.get("center", 0.0))
        return float(eval_lfo(shape, rate, depth, phase, center, t))

    if "noise" in mod:
        o = mod["noise"]
        typ = str(o.get("type", "smooth"))
        rate = float(o.get("rate", 1.0))
        depth = float(o.get("depth", 0.0))
        seed_key = str(o.get("seedKey", "noise"))
        mode = "step" if typ == "step" else "smooth"
        return float(eval_value_noise(root_seed, node_id, param_path, op_version, "noise", seed_key, t, rate, depth, mode))

    if "drift" in mod:
        o = mod["drift"]
        rate = float(o.get("rate", 0.25))
        depth = float(o.get("depth", 0.0))
        seed_key = str(o.get("seedKey", "drift"))
        return float(eval_value_noise(root_seed, node_id, param_path, op_version, "drift", seed_key, t, rate, depth, "smooth"))

    if "jitter" in mod:
        o = mod["jitter"]
        rate = float(o.get("rate", 6.0))
        depth = float(o.get("depth", 0.0))
        seed_key = str(o.get("seedKey", "jitter"))
        # highpass-ish: difference of smooth noise across ~1 frame
        n1 = eval_value_noise(root_seed, node_id, param_path, op_version, "jitter", seed_key, t, rate, 1.0, "smooth")
        n0 = eval_value_noise(root_seed, node_id, param_path, op_version, "jitter", seed_key, max(0.0, t - frame_dt), rate, 1.0, "smooth")
        return float(depth * (n1 - n0))

    return 0.0


def eval_modstack(modstack: dict,
                  base_value: Any,
                  macros: Dict[str, float],
                  root_seed: int,
                  node_id: str,
                  param_path: str,
                  op_version: int,
                  t: float,
                  frame_dt: float) -> Any:
    """
    Applies ModStack to a numeric base value.
    """
    if not isinstance(base_value, (int, float, np.floating, np.integer)):
        return base_value

    v = float(base_value)

    stack = modstack.get("stack", [])
    mode = modstack.get("mode", "add")
    clamp_v = modstack.get("clamp", None)

    # scaleByMacro template (closed)
    scale = 1.0
    sbm = modstack.get("scaleByMacro", None)
    if isinstance(sbm, dict):
        # supports:
        # { "motion": {curve, amount}, "complexity": {curve, amount} }
        for name, cfg in sbm.items():
            m = clamp01(float(macros.get(name, 0.0)))
            curve = str(cfg.get("curve", "linear"))
            amount = float(cfg.get("amount", 1.0))
            scale += amount * curve_eval(curve, m)

    delta = 0.0
    for mod in stack:
        delta += eval_modulator(mod, macros, root_seed, node_id, param_path, op_version, t, frame_dt)

    delta *= scale

    if mode == "add":
        out = v + delta
    elif mode == "mul":
        out = v * (1.0 + delta)
    elif mode == "replace":
        out = delta
    else:
        out = v + delta

    if clamp_v is not None:
        lo, hi = float(clamp_v[0]), float(clamp_v[1])
        out = max(lo, min(hi, out))

    return float(out)


# ----------------------------
# perFramePack: sampleLike
# ----------------------------

def speed_profile(speed: str) -> Tuple[float, float, float]:
    # drift_rate, jitter_rate, lfo_rate
    s = speed.lower()
    if s == "fast":
        return 0.55, 8.0, 0.15
    if s == "mid":
        return 0.30, 6.0, 0.10
    return 0.18, 4.0, 0.07  # slow


def expand_perframe_pack(node: dict,
                         macros: Dict[str, float],
                         root_seed: int,
                         op_versions: Dict[str, int]) -> Dict[str, dict]:
    """
    Returns additional perFrame entries to merge (paramPath -> ModStack).
    Deterministic band picking for spectralMask.
    """
    pack = node.get("perFramePack", None)
    if not pack:
        return {}

    if isinstance(pack, dict) and pack.get("name") != "sampleLike":
        return {}

    intensity = float(pack.get("intensity", 0.5))
    speed = str(pack.get("speed", "slow"))
    drift_rate, jitter_rate, lfo_rate = speed_profile(speed)

    op = node.get("op", "")
    node_id = node.get("id", "")
    opv = int(op_versions.get(op, 1))
    out: Dict[str, dict] = {}

    # scale by macro motion (global)
    motion = clamp01(float(macros.get("motion", 0.0)))
    complexity = clamp01(float(macros.get("complexity", 0.5)))

    # Common depth scalars
    motion_scale = 0.4 + 0.8 * motion  # 0.4..1.2
    comp_scale = 0.7 + 1.0 * complexity  # 0.7..1.7
    I = intensity * motion_scale

    if op == "formantBank":
        # shift drift, q jitter, optional gentle lfo on amount
        out["/shift"] = {
            "stack": [
                {"drift": {"rate": drift_rate, "depth": 0.10 * I, "seedKey": "shift"}},
            ],
            "mode": "add",
            "clamp": [-1.0, 1.0]
        }
        out["/q"] = {
            "stack": [
                {"jitter": {"rate": jitter_rate, "depth": 1.2 * I, "seedKey": "q"}},
            ],
            "mode": "add",
            "clamp": [0.5, 40.0]
        }
        out["/amount"] = {
            "stack": [
                {"lfo": {"shape": "sine", "rate": lfo_rate, "depth": 0.12 * I, "phase": 0.1, "center": 0.0}},
            ],
            "mode": "add",
            "clamp": [0.0, 2.0]
        }
        return out

    if op == "spectralMask":
        # pick a few db[k] bands deterministically
        p = node.get("p", {})
        bands = int(p.get("bands", 48))
        k_pick = 4 if bands >= 16 else max(1, bands // 4)

        # derive deterministic picks
        s = seed_sub(root_seed, [node_id, "perFramePack", "sampleLike", "bandPick", opv])
        rng = PCG32(seed=s)
        picks = set()
        while len(picks) < k_pick and bands > 0:
            idx = int(rng.random_u32() % max(1, bands))
            picks.add(idx)
        picks = sorted(list(picks))

        for idx in picks:
            ptr = f"/db/{idx}"
            out[ptr] = {
                "stack": [
                    {"noise": {"type": "smooth", "rate": 1.8 + 2.0 * complexity, "depth": 2.0 * I * comp_scale, "seedKey": f"db{idx}"}},
                ],
                "mode": "add",
                "clamp": [-60.0, 12.0]
            }
        return out

    if op == "tilt":
        # gentle drift on dbPerOct
        out["/dbPerOct"] = {
            "stack": [
                {"drift": {"rate": drift_rate * 0.7, "depth": 2.0 * I, "seedKey": "tilt"}},
            ],
            "mode": "add",
            "clamp": [-36.0, 36.0]
        }
        return out

    return {}


# ----------------------------
# spectralData decode + reconstruction (minimum phase IFFT)
# ----------------------------

@dataclass
class SpectralPayload:
    table_size: int
    frames: int
    harm_count: int
    noise_bands: int
    harm_amp: np.ndarray   # (F,H) amplitude in exporter units (amp = (2/N)*|X|)
    noise_db: np.ndarray   # (F,B) dB in 0.5 step range
    transient_q15: np.ndarray  # (F,3) (amount, center, width) placeholders


def decode_harm_noise_framepack(b64: str) -> SpectralPayload:
    raw = base64.b64decode(b64.encode("ascii"))
    if len(raw) < 8 + 8:
        raise WTGenError("spectralData payload too small")

    magic = raw[:8]
    if magic != b"HNFPv1\0":
        raise WTGenError(f"Bad spectralData magic: {magic!r}")

    table_size, F, H, B = struct.unpack_from("<HHHH", raw, 8)
    offset = 8 + 8

    per_frame_bytes = (H * 2) + (B * 2) + (3 * 2)
    need = offset + F * per_frame_bytes
    if len(raw) < need:
        raise WTGenError("spectralData payload truncated")

    harm_q = np.zeros((F, H), dtype=np.uint16)
    noise_q = np.zeros((F, B), dtype=np.int16)
    trans = np.zeros((F, 3), dtype=np.uint16)

    for i in range(F):
        hbytes = raw[offset:offset + H * 2]; offset += H * 2
        nbytes = raw[offset:offset + B * 2]; offset += B * 2
        tbytes = raw[offset:offset + 6]; offset += 6
        harm_q[i] = np.frombuffer(hbytes, dtype="<u2", count=H)
        noise_q[i] = np.frombuffer(nbytes, dtype="<i2", count=B)
        trans[i] = np.frombuffer(tbytes, dtype="<u2", count=3)

    harm_amp = harm_q.astype(np.float64) / 4096.0   # u16_q12 -> amplitude units
    noise_db = noise_q.astype(np.float64) / 2.0     # 0.5 dB steps
    return SpectralPayload(
        table_size=int(table_size),
        frames=int(F),
        harm_count=int(H),
        noise_bands=int(B),
        harm_amp=harm_amp,
        noise_db=noise_db,
        transient_q15=trans
    )


def band_edges_for_mask(N: int, bands: int, lo_bin: int = 1) -> List[Tuple[int, int]]:
    hi_bin = N // 2
    return _linear_band_edges(lo_bin, hi_bin, bands)


def magnitude_to_minimum_phase_rfft(mag_rfft: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Given rfft magnitudes for length-N real signal (size N//2+1),
    returns complex rfft spectrum with minimum phase.
    """
    # Build full spectrum magnitudes (two-sided)
    r = mag_rfft.astype(np.float64, copy=False)
    N2 = (len(r) - 1) * 2
    if N2 <= 0:
        return np.zeros_like(mag_rfft, dtype=np.complex128)

    # full mags
    mag_full = np.zeros((N2,), dtype=np.float64)
    mag_full[:len(r)] = r
    # mirror excluding DC and Nyq
    mag_full[len(r):] = r[-2:0:-1]

    logmag = np.log(np.maximum(mag_full, eps))
    cep = np.fft.ifft(logmag).real

    cep_min = np.zeros_like(cep)
    cep_min[0] = cep[0]
    if N2 % 2 == 0:
        # even
        cep_min[1:N2 // 2] = 2.0 * cep[1:N2 // 2]
        cep_min[N2 // 2] = cep[N2 // 2]
    else:
        cep_min[1:(N2 + 1) // 2] = 2.0 * cep[1:(N2 + 1) // 2]

    spec = np.fft.fft(cep_min)
    H = np.exp(spec)  # complex
    # return rfft slice
    return H[:len(r)].astype(np.complex128)


def spectraldata_reconstruct_frame(payload: SpectralPayload,
                                  frame_i: int,
                                  phase_mode: str = "minimumPhase") -> np.ndarray:
    """
    Returns complex rfft spectrum (N//2+1) for frame_i.
    """
    N = payload.table_size
    rlen = N // 2 + 1
    mag = np.zeros((rlen,), dtype=np.float64)

    # Harmonics: amplitude units -> |X| = amp * (N/2)
    amp = payload.harm_amp[frame_i]
    H = payload.harm_count
    for k in range(1, min(H, rlen - 1) + 1):
        mag[k] = float(amp[k - 1]) * (N / 2.0)

    # Residual/noise bands (constant amp within band)
    B = payload.noise_bands
    if B > 0:
        edges = _linear_band_edges(H + 1, N // 2, B)
        for b, (a, c) in enumerate(edges):
            db = float(payload.noise_db[frame_i, b])
            db = max(NOISE_DB_MIN, min(NOISE_DB_MAX, db))
            band_rms_amp = 10.0 ** (db / 20.0)  # amplitude units (same scale as harm_amp)
            bin_mag = band_rms_amp * (N / 2.0)
            mag[a:c + 1] = bin_mag

    # DC / Nyq left as 0
    if phase_mode == "minimumPhase":
        return magnitude_to_minimum_phase_rfft(mag)
    else:
        # deterministic zero phase
        return mag.astype(np.complex128)


# ----------------------------
# Ops (tilt, spectralMask, formantBank) on rfft spectra
# ----------------------------

# Simple vowel presets: (pos 0..1, gainDb, q)
VOWEL_PRESETS: Dict[str, List[Tuple[float, float, float]]] = {
    "ae": [(0.06, 10.0, 6.0), (0.18, 8.0, 10.0), (0.35, 6.0, 12.0)],
    "ou": [(0.04, 10.0, 7.0), (0.12, 7.0, 10.0), (0.25, 6.0, 12.0)],
    "ee": [(0.05, 8.0, 8.0), (0.22, 10.0, 12.0), (0.42, 7.0, 14.0)],
    "oo": [(0.03, 10.0, 7.0), (0.10, 7.0, 10.0), (0.20, 6.0, 12.0)],
    "ah": [(0.07, 10.0, 6.0), (0.20, 8.0, 10.0), (0.38, 6.0, 12.0)],
}

def op_tilt(spec: np.ndarray, db_per_oct: float, pivot_harm: int = 1) -> np.ndarray:
    out = spec.copy()
    N2 = (len(spec) - 1) * 2
    if N2 <= 0:
        return out
    rlen = len(spec)
    pivot = max(1, int(pivot_harm))
    for k in range(1, rlen):
        # treat bin k as "harmonic index k"
        rel = k / float(pivot)
        if rel <= 1e-12:
            g = 1.0
        else:
            gain_db = db_per_oct * math.log(rel, 2.0)
            g = 10.0 ** (gain_db / 20.0)
        out[k] *= g
    return out


def _smooth_array(x: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.0:
        return x
    # very simple 1D blur; amount in [0..1]
    # kernel size 3
    y = x.copy()
    a = float(amount)
    y[1:-1] = (1.0 - a) * x[1:-1] + 0.5 * a * (x[0:-2] + x[2:])
    y[0] = (1.0 - a) * x[0] + a * x[1]
    y[-1] = (1.0 - a) * x[-1] + a * x[-2]
    return y


def op_spectral_mask(spec: np.ndarray,
                     bands: int,
                     db: List[float],
                     smooth: float = 0.5,
                     amount: float = 1.0) -> np.ndarray:
    out = spec.copy()
    rlen = len(spec)
    bands = max(4, int(bands))
    if len(db) != bands:
        # pad/trim
        dd = list(db)[:bands] + [0.0] * max(0, bands - len(db))
    else:
        dd = list(db)

    gains = np.array([10.0 ** (float(v) / 20.0) for v in dd], dtype=np.float64)
    gains = _smooth_array(gains, float(smooth))

    edges = _linear_band_edges(1, (rlen - 1), bands)
    for b, (a, c) in enumerate(edges):
        g = float(gains[b])
        # mix gain toward 1 by "amount"
        gm = (1.0 - float(amount)) * 1.0 + float(amount) * g
        out[a:c + 1] *= gm
    return out


def op_formant_bank(spec: np.ndarray,
                    preset: str,
                    amount: float = 1.0,
                    q: float = 8.0,
                    shift: float = 0.0,
                    spread: float = 0.0) -> np.ndarray:
    out = spec.copy()
    rlen = len(spec)
    preset = str(preset)
    peaks = VOWEL_PRESETS.get(preset, VOWEL_PRESETS["ae"])

    # build envelope in linear gain per bin
    env = np.ones((rlen,), dtype=np.float64)
    # shift applied to pos in [-1..1] -> +/-0.08 of spectrum by default
    sh = float(shift) * 0.08
    sp = 1.0 + float(spread) * 0.8
    base_q = max(0.5, float(q))

    # Use gaussian on bin index (freq ~ k)
    for (pos, gain_db, q0) in peaks:
        cpos = clamp01(float(pos) + sh)
        center = 1.0 + cpos * (rlen - 2)
        # effective bandwidth
        qq = max(0.5, (base_q * float(q0) / 8.0) / sp)
        bw = max(1.0, center / qq)
        g_lin = 10.0 ** (float(gain_db) / 20.0)

        k = np.arange(rlen, dtype=np.float64)
        z = (k - center) / max(bw, 1e-9)
        bump = 1.0 + (g_lin - 1.0) * np.exp(-0.5 * z * z)
        env *= bump

    # mix envelope vs none by amount
    a = float(amount)
    env_mix = (1.0 - a) + a * env
    out *= env_mix.astype(np.float64)
    return out


# ----------------------------
# Graph loader + validator + topo sort
# ----------------------------

def validate_preset(doc: dict) -> None:
    if not isinstance(doc, dict):
        raise WTGenError("Preset must be a JSON object")
    if doc.get("schema") != "wtgen-1":
        raise WTGenError(f"Unsupported schema: {doc.get('schema')}")
    wt = doc.get("wt", {})
    if int(wt.get("channels", 1)) != 1:
        raise WTGenError("WTGEN-1 reference engine supports channels=1 only")
    prog = doc.get("program", {})
    if prog.get("mode") != "graph":
        raise WTGenError("This reference engine currently supports program.mode='graph' only")
    if "nodes" not in prog or "out" not in prog:
        raise WTGenError("program must contain nodes[] and out")
    nodes = prog["nodes"]
    if not isinstance(nodes, list) or not nodes:
        raise WTGenError("program.nodes must be a non-empty list")

    ids = set()
    for n in nodes:
        nid = n.get("id", None)
        if not isinstance(nid, str) or not nid:
            raise WTGenError("Each node must have a non-empty string id")
        if nid in ids:
            raise WTGenError(f"Duplicate node id: {nid}")
        ids.add(nid)

    out_id = prog.get("out")
    if out_id not in ids:
        raise WTGenError(f"program.out references unknown node id: {out_id}")


def topo_sort(nodes: List[dict]) -> List[dict]:
    # Kahn stable: prefer original order
    id_to_node = {n["id"]: n for n in nodes}
    indeg = {n["id"]: 0 for n in nodes}
    outs: Dict[str, List[str]] = {n["id"]: [] for n in nodes}

    for n in nodes:
        nid = n["id"]
        ins = n.get("in", [])
        if ins is None:
            ins = []
        if not isinstance(ins, list):
            raise WTGenError(f"Node {nid} 'in' must be list")
        for src in ins:
            if src not in id_to_node:
                raise WTGenError(f"Node {nid} input references unknown id: {src}")
            indeg[nid] += 1
            outs[src].append(nid)

    order_ids = [n["id"] for n in nodes]
    q = [nid for nid in order_ids if indeg[nid] == 0]
    out_list: List[str] = []

    while q:
        nid = q.pop(0)
        out_list.append(nid)
        for v in outs[nid]:
            indeg[v] -= 1
            if indeg[v] == 0:
                # stable insert by original order
                if v not in q:
                    q.append(v)
                    q.sort(key=lambda x: order_ids.index(x))

    if len(out_list) != len(nodes):
        raise WTGenError("Graph has cycles (WTGEN-1 requires DAG)")
    return [id_to_node[nid] for nid in out_list]


# ----------------------------
# Node param evaluation (MacroParam + perFrame + pack)
# ----------------------------

def deep_copy_json(x: Any) -> Any:
    return json.loads(json.dumps(x))


def evaluate_node_params(node: dict,
                         macros: Dict[str, float],
                         root_seed: int,
                         op_versions: Dict[str, int],
                         frame_i: int,
                         frames_total: int) -> dict:
    """
    Returns evaluated params dict for this node on this frame.
    Applies:
      - MacroParam.kind
      - perFrame ModStack (JSON Pointer)
    """
    op = node.get("op", "")
    node_id = node.get("id", "")
    opv = int(op_versions.get(op, 1))

    t = 0.0 if frames_total <= 1 else float(frame_i) / float(frames_total - 1)
    frame_dt = 0.0 if frames_total <= 1 else 1.0 / float(frames_total - 1)

    p = deep_copy_json(node.get("p", {}))

    # 1) Evaluate MacroParams recursively
    def walk(obj: Any, path_ptr: str) -> Any:
        if is_macro_param(obj):
            return eval_macro_param(obj, macros, root_seed, node_id, path_ptr, opv, t)
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                child_ptr = path_ptr + "/" + str(k).replace("~", "~0").replace("/", "~1")
                out[k] = walk(v, child_ptr)
            return out
        if isinstance(obj, list):
            outl = []
            for i, v in enumerate(obj):
                child_ptr = path_ptr + f"/{i}"
                outl.append(walk(v, child_ptr))
            return outl
        return obj

    p = walk(p, "")

    # 2) Merge perFramePack expansion into perFrame
    per_frame: Dict[str, dict] = node.get("perFrame", {}) or {}
    pack_add = expand_perframe_pack(node, macros, root_seed, op_versions)
    # merge (pack adds if missing, otherwise append stacks)
    per_frame_merged: Dict[str, dict] = deep_copy_json(per_frame)
    for ptr, ms in pack_add.items():
        if ptr not in per_frame_merged:
            per_frame_merged[ptr] = ms
        else:
            # merge stacks by concatenation
            a = per_frame_merged[ptr]
            if "stack" in a and "stack" in ms:
                a["stack"] = list(a.get("stack", [])) + list(ms.get("stack", []))
            per_frame_merged[ptr] = a

    # 3) Apply perFrame to matching pointers inside p
    # We apply only if pointer exists; otherwise ignore (compat).
    for ptr, modstack in per_frame_merged.items():
        try:
            base_val = json_get(p, ptr)
        except Exception:
            continue
        new_val = eval_modstack(modstack, base_val, macros, root_seed, node_id, ptr, opv, t, frame_dt)
        try:
            json_set(p, ptr, new_val)
        except Exception:
            continue

    return p


# ----------------------------
# Engine execution
# ----------------------------

def run_graph(doc: dict) -> np.ndarray:
    """
    Executes WTGEN graph and returns wavetable frames (F,N) float32 in [-1,1] after post.
    """
    validate_preset(doc)

    wt = doc["wt"]
    N = int(wt["tableSize"])
    F = int(wt["frames"])

    seed = int(doc.get("seed", 1))

    prog = doc["program"]
    macros = prog.get("macros", {}) or {}
    # defaults
    macros = {
        "complexity": clamp01(float(macros.get("complexity", 0.5))),
        "brightness": clamp01(float(macros.get("brightness", 0.5))),
        "motion": clamp01(float(macros.get("motion", 0.0))),
    }

    engine = doc.get("engine", {}) or {}
    op_versions = engine.get("opVersions", {}) or {}

    nodes = prog["nodes"]
    ordered_nodes = topo_sort(nodes)

    # Cache spectralData decoded payloads by node id
    spectral_payloads: Dict[str, SpectralPayload] = {}

    # output spectra for current frame
    node_spec: Dict[str, np.ndarray] = {}

    out_id = prog["out"]
    frames_time = np.zeros((F, N), dtype=np.float32)

    for i in range(F):
        node_spec.clear()
        for node in ordered_nodes:
            nid = node["id"]
            op = node.get("op", "")
            ins = node.get("in", []) or []

            # evaluate params for this frame (MacroParam + perFrame + pack)
            p_eval = evaluate_node_params(node, macros, seed, op_versions, i, F)

            if op == "spectralData":
                # decode payload once
                if nid not in spectral_payloads:
                    data_b64 = p_eval.get("data")
                    if not isinstance(data_b64, str):
                        raise WTGenError("spectralData missing data")
                    payload = decode_harm_noise_framepack(data_b64)
                    if payload.table_size != N or payload.frames != F:
                        raise WTGenError("spectralData payload mismatch with wt.tableSize/frames")
                    spectral_payloads[nid] = payload
                payload = spectral_payloads[nid]
                phase_mode = (p_eval.get("phase", {}) or {}).get("mode", "minimumPhase")
                spec = spectraldata_reconstruct_frame(payload, i, phase_mode=str(phase_mode))
                node_spec[nid] = spec
                continue

            # input handling: all ops here are 1-in spectral
            if len(ins) < 1:
                raise WTGenError(f"Op {op} requires input (node {nid})")
            src_id = ins[0]
            if src_id not in node_spec:
                raise WTGenError(f"Node {nid} missing evaluated input {src_id}")
            spec_in = node_spec[src_id]

            if op == "tilt":
                db_per_oct = float(p_eval.get("dbPerOct", 0.0))
                pivot = (p_eval.get("pivot", {}) or {}).get("harm", 1)
                node_spec[nid] = op_tilt(spec_in, db_per_oct=db_per_oct, pivot_harm=int(pivot))
                continue

            if op == "spectralMask":
                bands = int(p_eval.get("bands", 48))
                db = p_eval.get("db", [0.0] * bands)
                smooth = float(p_eval.get("smooth", 0.5))
                amount = float(p_eval.get("amount", 1.0))
                if not isinstance(db, list):
                    db = list(db)
                node_spec[nid] = op_spectral_mask(spec_in, bands=bands, db=db, smooth=smooth, amount=amount)
                continue

            if op == "formantBank":
                preset = str(p_eval.get("preset", "ae"))
                amount = float(p_eval.get("amount", 1.0))
                q = float(p_eval.get("q", 8.0))
                shift = float(p_eval.get("shift", 0.0))
                spread = float(p_eval.get("spread", 0.0))
                node_spec[nid] = op_formant_bank(spec_in, preset=preset, amount=amount, q=q, shift=shift, spread=spread)
                continue

            # compatibility: unknown op passthrough
            node_spec[nid] = spec_in

        # grab output spectrum
        spec_out = node_spec[out_id]
        # time domain
        frame = np.fft.irfft(spec_out, n=N).astype(np.float64)
        frames_time[i] = frame.astype(np.float32)

    # post (dcRemove + normalize)
    post = doc.get("post", {}) or {}
    if bool(post.get("dcRemove", False)):
        for i in range(F):
            frames_time[i] = (frames_time[i] - float(np.mean(frames_time[i]))).astype(np.float32)

    norm = post.get("normalize", None)
    if isinstance(norm, dict):
        mode = str(norm.get("mode", "peak"))
        target = float(norm.get("target", 0.999))
        scope = str(norm.get("scope", "global"))

        if scope == "frame":
            for i in range(F):
                x = frames_time[i].astype(np.float64)
                if mode == "rms":
                    r = float(np.sqrt(np.mean(x * x)))
                    g = (target / max(r, 1e-12))
                else:
                    p = float(np.max(np.abs(x)))
                    g = (target / max(p, 1e-12))
                frames_time[i] = (x * g).astype(np.float32)
        else:
            x = frames_time.astype(np.float64)
            if mode == "rms":
                r = float(np.sqrt(np.mean(x * x)))
                g = (target / max(r, 1e-12))
            else:
                p = float(np.max(np.abs(x)))
                g = (target / max(p, 1e-12))
            frames_time = (x * g).astype(np.float32)

    # final clamp
    frames_time = np.clip(frames_time, -1.0, 1.0).astype(np.float32)
    return frames_time


# ============================================================
# CLI: export + render
# ============================================================

def cmd_export(args):
    in_path = Path(args.input_wav).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    out_path = Path(args.output).expanduser().resolve() if args.output else in_path.with_suffix(".wtgen.json")

    frames, meta = wav_to_frames_and_meta(
        wav_path=in_path,
        sr_target=int(args.sr),
        frames_n=int(args.frames),
        table_size=int(args.tableSize)
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
        noise_bands=int(args.noiseBands)
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
    print(
        f"     spectralData: harmonics={min(int(args.harmonics), (int(args.tableSize)//2)-1)} "
        f"noiseBands={int(args.noiseBands)} phaseMode={PHASE_MODE} "
        f"quant=({HARM_QUANT}, {NOISE_DB_QUANT})"
    )


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


