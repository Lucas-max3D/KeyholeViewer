"""Batch auto-detection over all samples (runs in worker processes).

Each worker analyses one sample: acoustic RMS detection + video keyhole tracking. The full per-frame
video analysis is written to the same cache file the viewer uses, so opening a sample afterwards is
instant. A compact summary (events, stats, speed) is returned and collected in summary.json.
"""
from __future__ import annotations

import json
import os
import time

import data
import detect

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "annotations", "cache")
SUMMARY = os.path.join(CACHE_DIR, "summary.json")
DEFAULT_CFG = {"thr_V": 0.17, "thr_px": 12, "dwell_ms": 0.3, "video_thr": 0.07, "step": 2}


def video_cache_path(key, v, thr, step):
    return os.path.join(CACHE_DIR, f"{key}_video_v{detect.VERSION}_thr{thr:g}_step{step}_last{v.last}.json")


def laser_window(s: dict):
    """(gate, assumed): measured gate from acoustics, else the window implied by the sheet."""
    gate = None
    if s.get("acoustic_file"):
        try:
            gate = data.laser_from_file(s["acoustic_file"])
        except Exception:
            gate = None
    if gate:
        return tuple(float(x) for x in gate), False
    pi = data.process_info(s.get("params"), None)
    if pi and pi.get("laser_ms"):
        return tuple(pi["laser_ms"]), True
    return None, False


def analyse_sample(s: dict, cfg: dict | None = None, force=False) -> dict:
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    t0 = time.time()
    out = {"key": s["key"], "id": s["id"], "material": s["material"], "wall": s["wall"], "side": s["side"],
           "line": s["line"], "cfg": cfg, "acoustic": None, "video": None, "errors": []}
    laser, assumed = laser_window(s)
    out["laser"] = laser
    out["laser_assumed"] = assumed
    if s.get("acoustic_file"):
        try:
            ch = data.AcousticCache(maxfiles=1).get(s["acoustic_file"])
            if "RMS" not in ch:  # some files only recorded the laser gate (A2)
                out["acoustic_note"] = "laser gate (A2) only, no acoustic channel A1"
            else:
                a = detect.detect_acoustic(ch, None if assumed else laser, thr_V=cfg["thr_V"], min_dwell_ms=cfg["dwell_ms"])
                out["acoustic"] = {"events": a["events"], "open_ms": a["open_ms"]}
        except Exception as e:
            out["errors"].append(f"acoustic: {e!r}")
    if s.get("video_folder"):
        try:
            v = data.load_video(s["video_folder"])
            if v is None or v.last < v.first + 50:
                raise RuntimeError("video not (yet) available")
            path = video_cache_path(s["key"], v, cfg["video_thr"], cfg["step"])
            if os.path.exists(path) and not force:
                with open(path, encoding="utf-8") as f:
                    an = json.load(f)
            else:
                an = detect.analyze_video(v, laser, thr=cfg["video_thr"], step=cfg["step"])
                an["laser_assumed"] = assumed
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(an, f)
                os.replace(tmp, path)
            ev = detect.video_events(an, thr_px=cfg["thr_px"], min_dwell_ms=cfg["dwell_ms"], laser=laser)
            out["video"] = {"events": ev, "stats": detect.open_stats(an, ev), "speed": detect.scan_speed(an),
                            "surface_row": an["surface_row"], "scan_dir": an["scan_dir"],
                            "last_frame_ms": v.last / v.meta["fps"] * 1e3}
        except Exception as e:
            out["errors"].append(f"video: {e!r}")
    out["seconds"] = round(time.time() - t0, 1)
    return out


def load_summary() -> dict:
    if os.path.exists(SUMMARY):
        try:
            with open(SUMMARY, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_summary(d: dict):
    tmp = SUMMARY + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
    os.replace(tmp, SUMMARY)
