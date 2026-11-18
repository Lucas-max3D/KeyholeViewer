"""Data indexing and readers for the ID19 beamtime (Ti64 + 316L).

- Videos: Photron FASTCAM SA-Z multi-page TIFF split in several files ("-00.tif", "-01.tif", ...),
  each with a .cihx metadata file (frame numbers relative to the trigger, frame rate).
- Acoustics: Elsys TranAX .tpc5 (HDF5) with two channels:
    A1 = acoustic emission (microphone), A2 = laser gate signal.
  Time 0 = trigger sample.

Video frame n (relative to trigger) is at t = n / fps (+ optional per-sample sync offset).
"""
from __future__ import annotations

import os
import re
import threading
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None
import tifffile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Folders that only contain vendor examples -> never scanned
EXCLUDE_DIRS = {"HighAPI", "Python_ReadTpc5_Rev12_Examples", "MatlabDataImportExample",
                "HardwareDesign", "Images", "KeyholeViewer"}
CALIB_RE = re.compile(r"(flat|dark)", re.I)
# Samples left out of the index: (material, regex on the sample ID)
EXCLUDE_SAMPLES = [("316L", re.compile(r"^W[5-8]S", re.I))]
ID_RE = re.compile(r"^(W(\d+)(?:S(\d+))?L(\d+)(?:T(\d+))?)", re.I)


def material_from_path(path: str) -> str:
    p = path.lower()
    if "titan" in p or "ti64" in p or "\\ti" in p:
        return "Ti64"
    if "steel" in p or "316" in p:
        return "316L"
    return "Unknown"


def is_excluded(material: str, sid: str) -> bool:
    return any(m == material and rx.match(sid) for m, rx in EXCLUDE_SAMPLES)


def parse_id(name: str):
    m = ID_RE.match(name)
    if not m:
        return None
    return {
        "id": m.group(1).upper(),
        "wall": int(m.group(2)),
        "side": int(m.group(3)) if m.group(3) else None,
        "line": int(m.group(4)),
        "trial": int(m.group(5)) if m.group(5) else None,
    }


# ----------------------------------------------------------------------------------------- video
def read_cihx(path: str) -> dict:
    raw = open(path, "rb").read()
    i = raw.find(b"<?xml")
    j = raw.rfind(b"</cih>")
    root = ET.fromstring(raw[i:j + 6])

    def g(p, cast=int, default=None):
        el = root.find(p)
        if el is None or el.text is None:
            return default
        try:
            return cast(el.text)
        except ValueError:
            return default

    return {
        "width": g("imageDataInfo/resolution/width"),
        "height": g("imageDataInfo/resolution/height"),
        "bit": g("imageDataInfo/colorInfo/bit"),
        "effectiveBit": g("imageDataInfo/effectiveBit/depth"),
        "fps": g("recordInfo/recordRate", float),
        "startFrame": g("frameInfo/startFrame"),
        "totalFrame": g("frameInfo/totalFrame"),
        "saveStart": g("frameInfo/saveStartFrameNo"),
        "saveEnd": g("frameInfo/saveEndFrameNo"),
        "shutterNs": g("recordInfo/shutterSpeedNsec"),
        "date": g("fileInfo/date", str),
        "time": g("fileInfo/time", str),
        "camera": g("deviceInfo/deviceName", str),
    }


@dataclass
class VideoPart:
    tif: str
    start: int          # first frame number (relative to trigger)
    count: int          # frames declared in cihx
    offset0: int = 0    # byte offset of frame 0 pixel data
    stride: int = 0     # bytes between frames
    available: int = 0  # frames physically present (file may still be copying)

    def probe(self, w, h):
        self.available = 0
        if not os.path.exists(self.tif):
            return
        try:
            with tifffile.TiffFile(self.tif) as tf:
                p0 = tf.pages[0]
                if p0.compression != 1:
                    raise ValueError("compressed TIFF not supported")
                self.offset0 = int(p0.dataoffsets[0])
                if self.count > 1:
                    self.stride = int(tf.pages[1].dataoffsets[0]) - self.offset0
                else:
                    self.stride = w * h * 2
        except Exception:
            return
        size = os.path.getsize(self.tif)
        n = max(0, (size - self.offset0 - w * h * 2) // self.stride + 1) if self.stride else 0
        self.available = int(min(n, self.count))


@dataclass
class Video:
    folder: str
    name: str
    meta: dict
    parts: list = field(default_factory=list)

    @property
    def first(self):
        return min(p.start for p in self.parts)

    @property
    def last(self):
        # last frame physically available, contiguous from the first part
        last = self.first - 1
        for p in sorted(self.parts, key=lambda p: p.start):
            if p.start != last + 1:
                break
            last = p.start + p.available - 1
            if p.available < p.count:
                break
        return last

    @property
    def declared_last(self):
        return max(p.start + p.count - 1 for p in self.parts)

    def read_frame(self, n: int) -> np.ndarray:
        w, h = self.meta["width"], self.meta["height"]
        for p in self.parts:
            if p.start <= n < p.start + p.available:
                k = n - p.start
                with open(p.tif, "rb") as fh:
                    fh.seek(p.offset0 + k * p.stride)
                    buf = fh.read(w * h * 2)
                return np.frombuffer(buf, dtype="<u2").reshape(h, w)
        raise IndexError(f"frame {n} not available")

    def read_rows(self, n: int, y0: int, y1: int) -> np.ndarray:
        """Only rows y0..y1 of frame n (rows are contiguous on disk -> less I/O)."""
        w = self.meta["width"]
        for p in self.parts:
            if p.start <= n < p.start + p.available:
                k = n - p.start
                with open(p.tif, "rb") as fh:
                    fh.seek(p.offset0 + k * p.stride + y0 * w * 2)
                    buf = fh.read((y1 - y0) * w * 2)
                return np.frombuffer(buf, dtype="<u2").reshape(y1 - y0, w)
        raise IndexError(f"frame {n} not available")


def load_video(folder: str) -> Video | None:
    cihx = sorted(f for f in os.listdir(folder) if f.lower().endswith(".cihx"))
    if not cihx:
        return None
    parts, meta = [], None
    for c in cihx:
        m = read_cihx(os.path.join(folder, c))
        meta = meta or m
        tif = os.path.join(folder, c[:-5] + ".tif")
        parts.append(VideoPart(tif=tif, start=m["startFrame"], count=m["totalFrame"]))
    for p in parts:
        p.probe(meta["width"], meta["height"])
    return Video(folder=folder, name=os.path.basename(folder), meta=meta, parts=parts)


# -------------------------------------------------------------------------------------- acoustic
CH_BASE = "measurements/00000001/channels/"


def acoustic_header(path: str) -> dict:
    with h5py.File(path, "r") as f:
        chans = []
        for c in sorted(f[CH_BASE].keys()):
            g = f[CH_BASE + c]
            b = g["blocks/00000001"]
            chans.append({
                "key": c,
                "name": str(g.attrs.get("ChannelName", c)),
                "unit": str(g.attrs.get("physicalUnit", "V")),
                "fs": float(b.attrs["sampleRateHertz"]),
                "triggerSample": int(b.attrs["triggerSample"]),
                "n": int(b["raw"].shape[0]),
                "startTime": str(b.attrs.get("startTime", "")),
            })
    return {"file": path, "channels": chans}


class AcousticCache:
    """Keeps decoded channels of recently used files in memory (float32)."""

    def __init__(self, maxfiles=4):
        self.maxfiles = maxfiles
        self._c: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, path: str) -> dict:
        with self._lock:
            if path in self._c:
                self._c.move_to_end(path)
                return self._c[path]
        d = {}
        with h5py.File(path, "r") as f:
            for c in sorted(f[CH_BASE].keys()):
                g = f[CH_BASE + c]
                b = g["blocks/00000001"]
                raw = b["raw"][:] & int(g.attrs.get("analogMask", 65535))
                v = raw.astype(np.float32) * np.float32(g.attrs["binToVoltFactor"]) + np.float32(g.attrs["binToVoltConstant"])
                v = v * np.float32(g.attrs.get("voltToPhysicalFactor", 1.0)) + np.float32(g.attrs.get("voltToPhysicalConstant", 0.0))
                name = str(g.attrs.get("ChannelName", c))
                d[name.split(" ")[0]] = {
                    "y": v, "fs": float(b.attrs["sampleRateHertz"]),
                    "trig": int(b.attrs["triggerSample"]), "name": name,
                }
        # derived: moving RMS of high-passed A1 (acoustic emission envelope)
        if "A1" in d:
            a = d["A1"]
            y = a["y"].astype(np.float64)
            fs = a["fs"]
            # remove slow baseline (2 ms moving mean) then RMS over 50 us
            w0 = max(1, int(fs * 2e-3))
            base = _movmean(y, w0)
            hp = y - base
            w = max(1, int(fs * 50e-6))
            rms = np.sqrt(np.maximum(_movmean(hp * hp, w), 0)).astype(np.float32)
            d["RMS"] = {"y": rms, "fs": fs, "trig": a["trig"], "name": "A1 RMS (50 µs)"}
        with self._lock:
            self._c[path] = d
            while len(self._c) > self.maxfiles:
                self._c.popitem(last=False)
        return d


def _movmean(y: np.ndarray, w: int) -> np.ndarray:
    c = np.cumsum(np.concatenate([[0.0], y]))
    out = np.empty_like(y)
    h = w // 2
    n = y.size
    idx = np.arange(n)
    lo = np.clip(idx - h, 0, n)
    hi = np.clip(idx - h + w, 0, n)
    out[:] = (c[hi] - c[lo]) / np.maximum(hi - lo, 1)
    return out


def decimate_minmax(ch: dict, t0_ms: float, t1_ms: float, npix: int):
    """Return time (ms) and min/max per bucket for the window [t0, t1]."""
    y, fs, trig = ch["y"], ch["fs"], ch["trig"]
    i0 = int(np.floor(t0_ms * 1e-3 * fs)) + trig
    i1 = int(np.ceil(t1_ms * 1e-3 * fs)) + trig
    i0 = max(0, i0)
    i1 = min(y.size, i1)
    if i1 <= i0:
        return {"t": [], "min": [], "max": [], "raw": False}
    n = i1 - i0
    if n <= npix * 2:
        t = (np.arange(i0, i1) - trig) / fs * 1e3
        return {"t": t.round(6).tolist(), "y": y[i0:i1].round(5).tolist(), "raw": True}
    k = int(np.ceil(n / npix))
    m = n // k
    seg = y[i0:i0 + m * k].reshape(m, k)
    t = (i0 + np.arange(m) * k + k / 2 - trig) / fs * 1e3
    return {"t": t.round(6).tolist(), "min": seg.min(1).round(5).tolist(),
            "max": seg.max(1).round(5).tolist(), "raw": False}


# --------------------------------------------------------------------------------------- params
LINE_LENGTH_MM = 4.0
PARAMS_FILE = os.path.join(ROOT, "LinkWallNumberParameters.xlsx")


def sheet_material(title: str, n_sheets: int):
    t = title.lower()
    if "steel" in t or "316" in t or t.startswith("ss"):
        return "316L"
    if "ti" in t:
        return "Ti64"
    if n_sheets == 1:  # the original single-sheet file was the Ti64 table
        return "Ti64"
    return None


def _num(v):
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    try:
        return float(str(v).replace(",", "."))
    except ValueError:
        return None


def _key(wall, side, line):
    return (int(wall), int(side) if side not in (None, "") else 0, int(line))


def load_params() -> dict:
    """Read every sheet of LinkWallNumberParameters.xlsx -> {material: {(wall, side|0, line): rec}}.
    The material comes from the sheet name (steel/316 -> 316L, ti -> Ti64)."""
    out: dict = {}
    if not os.path.exists(PARAMS_FILE):
        return out
    try:
        import openpyxl
        wb = openpyxl.load_workbook(PARAMS_FILE, read_only=True, data_only=True)
        for ws in wb.worksheets:
            mat = sheet_material(ws.title, len(wb.worksheets))
            if mat is None:
                print(f"params: sheet {ws.title!r}: cannot tell the material from its name, skipped")
                continue
            rows = list(ws.iter_rows(values_only=True))
            hdr_i = next((i for i, r in enumerate(rows)
                          if r and "wall" in [str(c).strip().lower() for c in r if c is not None]), None)
            if hdr_i is None:
                continue
            hdr = [str(c).strip() if c is not None else "" for c in rows[hdr_i]]

            def col(rec, *names):
                for nm in names:
                    for h in hdr:
                        if h and h.lower().startswith(nm):
                            return rec.get(h)
                return None
            for r in rows[hdr_i + 1:]:
                rec = dict(zip(hdr, r))
                try:
                    key = _key(col(rec, "wall"), col(rec, "side"), col(rec, "line"))
                except (TypeError, ValueError):
                    continue
                d = {k: (str(v) if v is not None else "") for k, v in rec.items() if k}
                d["_power"] = col(rec, "power")
                d["_velocity"] = col(rec, "velocity", "speed")
                d["_profile"] = col(rec, "profile")
                d["_sheet"] = ws.title
                out.setdefault(mat, {})[key] = d
        wb.close()
    except Exception as e:  # file may be locked while open in Excel
        print("params:", e)
    return out


def power_profile(power, profile=None, length_mm=LINE_LENGTH_MM):
    """Power along the line as breakpoints (x_mm, P_W), linear in between.
    1 value: constant; 2 values: start -> end of the 4 mm line; 3 values: at 0, 2, 4 mm;
    a 'Profile' entry (mm, ';'-separated) overrides the positions."""
    if power is None or str(power).strip() == "":
        return None
    vals = [_num(v) for v in str(power).split(";")]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    note = ""
    pos = None
    if profile is not None and str(profile).strip():
        pos = [_num(v) for v in str(profile).split(";")]
        if None in pos or len(pos) != len(vals):
            note = f"profile '{profile}' does not match power '{power}': used equal spacing"
            pos = None
    if pos is None:
        pos = [0.0] if len(vals) == 1 else list(np.linspace(0, length_mm, len(vals)))
    text = " → ".join(f"{v:g}" for v in vals) + " W"
    if len(vals) > 1:
        text += " at " + " / ".join(f"{p:g}" for p in pos) + " mm"
    if len(vals) == 1:
        pos, vals = [0.0, length_mm], [vals[0], vals[0]]
    return {"x_mm": [round(float(p), 4) for p in pos], "P_W": [float(v) for v in vals], "note": note, "text": text}


def process_info(params, laser=None):
    """Combine sheet parameters with the laser gate -> position and power as a function of time.
    laser = (t_on, t_off) ms from the acoustics; without it the camera trigger (t = 0) is taken as laser on."""
    if not params:
        return None
    prof = power_profile(params.get("_power"), params.get("_profile"))
    v = _num(params.get("_velocity"))
    info = {"profile": prof, "velocity_mm_s": v, "v_source": "sheet" if v else None,
            "line_mm": LINE_LENGTH_MM, "notes": [], "sheet": params.get("_sheet")}
    if laser:
        t_on, t_off = laser
        dur = t_off - t_on
        info["laser_ms"] = [t_on, t_off]
        v_gate = LINE_LENGTH_MM / dur * 1e3 if dur > 0 else None
        info["v_from_gate_mm_s"] = round(v_gate, 2) if v_gate else None
        if v and v_gate and abs(v_gate / v - 1) > 0.1:
            info["notes"].append(f"laser on for {dur:.2f} ms, but {LINE_LENGTH_MM:g} mm at {v:g} mm/s takes "
                                 f"{LINE_LENGTH_MM / v * 1e3:.2f} ms (gate implies {v_gate:.0f} mm/s, "
                                 f"or a {v * dur / 1e3:.2f} mm line)")
        if not v and v_gate:
            info["velocity_mm_s"], info["v_source"] = round(v_gate, 2), "laser gate (4 mm / laser-on time)"
    else:
        info["laser_ms"] = [0.0, LINE_LENGTH_MM / v * 1e3] if v else None
        info["notes"].append("no acoustic laser gate: laser on assumed at the camera trigger (t = 0)")
    if prof and prof["note"]:
        info["notes"].append(prof["note"])
    if params.get("_assumed_side"):
        info["notes"].append(f"no side in the sample name: used sheet row side {params['_assumed_side']}")
    if prof is None:
        info["notes"].append("no power given in the sheet")
    return info


def power_at(info, t_ms):
    """(x_mm, P_W) at time t_ms; (None, None) if unknown, (None, 0) outside the laser-on window."""
    if not info or not info.get("profile") or not info.get("velocity_mm_s") or not info.get("laser_ms"):
        return None, None
    t_on, t_off = info["laser_ms"]
    if t_ms < t_on or t_ms > t_off + 1e-6:
        return None, 0.0
    x = (t_ms - t_on) * info["velocity_mm_s"] / 1e3
    p = float(np.interp(x, info["profile"]["x_mm"], info["profile"]["P_W"]))
    return round(x, 4), round(p, 2)


def laser_from_file(path, level=2.5):
    """Laser gate (t_on, t_off) in ms read from the A2 channel only (fast)."""
    with h5py.File(path, "r") as f:
        for c in f[CH_BASE].keys():
            g = f[CH_BASE + c]
            if not str(g.attrs.get("ChannelName", "")).startswith("A2"):
                continue
            b = g["blocks/00000001"]
            v = b["raw"][:].astype(np.float32) * np.float32(g.attrs["binToVoltFactor"]) + np.float32(g.attrs["binToVoltConstant"])
            on = np.flatnonzero(v > level)
            if on.size == 0:
                return None
            fs, tr = float(b.attrs["sampleRateHertz"]), int(b.attrs["triggerSample"])
            return (round((on[0] - tr) / fs * 1e3, 5), round((on[-1] - tr) / fs * 1e3, 5))
    return None


# ---------------------------------------------------------------------------------------- index
def scan(root: str = ROOT) -> dict:
    """Walk the beamtime folder and build {key: sample} merging videos and acoustics by ID."""
    samples: dict[str, dict] = {}
    calib: list[dict] = []
    params = load_params()

    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        top = rel.split(os.sep)[0]
        if rel != "." and (top in EXCLUDE_DIRS or top.startswith(".")):
            dirnames[:] = []
            continue
        base = os.path.basename(dirpath)
        mat = material_from_path(rel)

        if any(f.lower().endswith(".cihx") for f in filenames):
            if CALIB_RE.search(base):
                calib.append({"folder": dirpath, "name": base, "material": mat})
                continue
            pid = parse_id(base)
            if not pid or is_excluded(mat, pid["id"]):
                continue
            stamp = base[len(pid["id"]):].strip("_")
            key = f"{mat}_{pid['id']}"
            if key in samples and samples[key].get("video_folder"):
                key = f"{mat}_{pid['id']}_{stamp}"
            s = samples.setdefault(key, {"key": key, "material": mat, **pid})
            s["video_folder"] = dirpath
            s["video_stamp"] = stamp

        for fn in filenames:
            if fn.lower().endswith(".tpc5"):
                pid = parse_id(os.path.splitext(fn)[0])
                if not pid or is_excluded(mat, pid["id"]):
                    continue
                key = f"{mat}_{pid['id']}"
                s = samples.setdefault(key, {"key": key, "material": mat, **pid})
                s["acoustic_file"] = os.path.join(dirpath, fn)

    for s in samples.values():
        tab = params.get(s["material"], {})
        rec = tab.get(_key(s["wall"], s["side"], s["line"]))
        if rec is None and not s["side"]:
            # names like W1L4 carry no side; the sheet lists them as side 1
            rec = tab.get(_key(s["wall"], 1, s["line"]))
            if rec is not None:
                rec = dict(rec, _assumed_side=1)
        s["params"] = rec or {}
    return {"samples": samples, "calib": calib}
