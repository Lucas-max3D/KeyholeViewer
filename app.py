"""Keyhole Viewer – synchronized X-ray video + acoustic emission browser for the ID19 beamtime.

Run:  python app.py      then open http://127.0.0.1:5019
"""
from __future__ import annotations

import csv
import io
import json
import os
import threading
import time

import numpy as np
from scipy import ndimage as ndi
from flask import Flask, Response, abort, jsonify, request, send_from_directory

import batch
import data
import detect
import hysteresis

HERE = os.path.dirname(os.path.abspath(__file__))
ANN_DIR = os.path.join(HERE, "annotations")
os.makedirs(ANN_DIR, exist_ok=True)

app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="/static")

_index = {"samples": {}, "calib": [], "t": 0}
_videos: dict[str, data.Video] = {}
_acoustic_hdr: dict[str, dict] = {}
_acache = data.AcousticCache()
_lock = threading.Lock()


def rescan():
    idx = data.scan()
    with _lock:
        _index.update(idx)
        _index["t"] = time.time()
        _videos.clear()
        _acoustic_hdr.clear()


def sample(key) -> dict:
    s = _index["samples"].get(key)
    if not s:
        abort(404, f"unknown sample {key}")
    return s


def video(key) -> data.Video | None:
    s = sample(key)
    if not s.get("video_folder"):
        return None
    v = _videos.get(key)
    if v is None:
        v = data.load_video(s["video_folder"])
        _videos[key] = v
    return v


def acoustic_hdr(key):
    s = sample(key)
    if not s.get("acoustic_file"):
        return None
    h = _acoustic_hdr.get(key)
    if h is None:
        try:
            h = data.acoustic_header(s["acoustic_file"])
        except Exception as e:
            h = {"error": str(e)}
        _acoustic_hdr[key] = h
    return h


_laser_cache: dict = {}


def laser_gate(key):
    """Laser on/off (ms) from the acoustic A2 channel, cached; None without acoustics."""
    s = sample(key)
    f = s.get("acoustic_file")
    if not f:
        return None
    if f not in _laser_cache:
        try:
            _laser_cache[f] = data.laser_from_file(f)
        except Exception:
            _laser_cache[f] = None
    return _laser_cache[f]


def process(key):
    s = sample(key)
    return data.process_info(s.get("params"), laser_gate(key))


# ---------------------------------------------------------------------------------- annotations
def ann_path(key):
    return os.path.join(ANN_DIR, f"{key}.json")


def load_ann(key):
    p = ann_path(key)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {"key": key, "sync_offset_ms": 0.0, "events": [], "notes": ""}


def save_ann(key, a):
    a["key"] = key
    a["modified"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = ann_path(key) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(a, f, indent=2)
    os.replace(tmp, ann_path(key))


# ------------------------------------------------------------------------------------------ api
@app.route("/")
def root():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/samples")
def api_samples():
    if request.args.get("refresh") or not _index["samples"]:
        rescan()
    out = []
    for k, s in sorted(_index["samples"].items(),
                       key=lambda kv: (kv[1]["material"], kv[1]["wall"], kv[1]["side"] or 0, kv[1]["line"], kv[0])):
        n_ev = len(load_ann(k)["events"]) if os.path.exists(ann_path(k)) else 0
        out.append({
            "key": k, "id": s["id"], "material": s["material"], "wall": s["wall"], "side": s["side"],
            "line": s["line"], "trial": s["trial"], "hasVideo": bool(s.get("video_folder")),
            "hasAcoustic": bool(s.get("acoustic_file")), "events": n_ev, "params": s.get("params", {}),
        })
    return jsonify({"samples": out, "calib": [c["name"] for c in _index["calib"]]})


@app.route("/api/sample/<key>")
def api_sample(key):
    s = sample(key)
    info = {"key": key, "id": s["id"], "material": s["material"], "params": s.get("params", {}),
            "video": None, "acoustic": None}
    v = video(key)
    if v:
        # refresh availability (files may still be copying)
        for p in v.parts:
            if p.available < p.count:
                p.probe(v.meta["width"], v.meta["height"])
        info["video"] = {
            "folder": os.path.relpath(v.folder, data.ROOT), "stamp": s.get("video_stamp"),
            "fps": v.meta["fps"], "width": v.meta["width"], "height": v.meta["height"],
            "effectiveBit": v.meta["effectiveBit"], "first": v.first, "last": v.last,
            "declaredLast": v.declared_last, "camera": v.meta["camera"],
            "date": v.meta["date"], "time": v.meta["time"], "shutterNs": v.meta["shutterNs"],
            "parts": [{"file": os.path.basename(p.tif), "start": p.start, "count": p.count,
                       "available": p.available} for p in v.parts],
            "dark": os.path.basename(s["dark_folder"]) if s.get("dark_folder") else None,
        }
    h = acoustic_hdr(key)
    if h:
        if "error" in h:
            info["acoustic"] = {"error": h["error"], "file": os.path.relpath(s["acoustic_file"], data.ROOT)}
        else:
            c = h["channels"][0]
            info["acoustic"] = {
                "file": os.path.relpath(h["file"], data.ROOT), "fs": c["fs"], "startTime": c["startTime"],
                "tmin_ms": -c["triggerSample"] / c["fs"] * 1e3,
                "tmax_ms": (c["n"] - c["triggerSample"]) / c["fs"] * 1e3,
                "channels": [ch["name"] for ch in h["channels"]],
            }
    info["process"] = process(key)
    info["annotations"] = load_ann(key)
    return jsonify(info)


@app.route("/api/acoustic/<key>")
def api_acoustic(key):
    s = sample(key)
    if not s.get("acoustic_file"):
        abort(404)
    d = _acache.get(s["acoustic_file"])
    t0 = float(request.args.get("t0", -10))
    t1 = float(request.args.get("t1", 60))
    npix = int(min(4000, max(50, int(request.args.get("n", 1200)))))
    out = {}
    for name in request.args.get("ch", "A1,A2,RMS").split(","):
        if name in d:
            out[name] = data.decimate_minmax(d[name], t0, t1, npix)
            out[name]["label"] = d[name]["name"]
    return jsonify(out)


@app.route("/api/acoustic_value/<key>")
def api_acoustic_value(key):
    """Values of all channels at time t (ms) – used for the readout under the cursor."""
    s = sample(key)
    d = _acache.get(s["acoustic_file"])
    t = float(request.args.get("t", 0))
    out = {}
    for name, ch in d.items():
        i = int(round(t * 1e-3 * ch["fs"])) + ch["trig"]
        out[name] = float(ch["y"][i]) if 0 <= i < ch["y"].size else None
    return jsonify(out)


def dark(key):
    """Mean dark frame for a sample's video (None if no dark-field recording was found)."""
    s = sample(key)
    return data.dark_frame(s["dark_folder"], CACHE_DIR) if s.get("dark_folder") else None


@app.route("/api/frame/<key>/<int(signed=True):n>")
def api_frame(key, n):
    """Raw frame n as little-endian uint16. Optional processing (then float32 is returned):
    dark=1 subtract the mean dark frame; avg=k mean of k frames centred on n (or, with from=1, the k frames
    starting at n: used for the reference); sigma=s Gaussian smoothing [px]."""
    v = video(key)
    if not v:
        abort(404)
    a = request.args
    k = max(1, min(201, int(a.get("avg", 1))))
    sigma = max(0.0, min(5.0, float(a.get("sigma", 0))))
    use_dark = a.get("dark") == "1"
    if a.get("from") == "1":
        n0 = max(v.first, min(n, v.last - k + 1))
    else:
        n0 = max(v.first, min(n - (k - 1) // 2, v.last - k + 1))
    ns = [m for m in range(n0, n0 + k) if v.first <= m <= v.last]
    try:
        if k == 1 and sigma == 0 and not use_dark:
            arr = v.read_frame(n)
        else:
            arr = np.zeros((v.meta["height"], v.meta["width"]), np.float32)
            for m in ns:
                arr += v.read_frame(m)
            arr /= len(ns)
            if use_dark:
                d = dark(key)
                if d is not None:
                    arr -= d
            if sigma > 0:
                arr = ndi.gaussian_filter(arr, sigma)
            arr = arr.astype("<f4")
    except IndexError:
        abort(404)
    step = int(a.get("bin", 1))
    if step > 1:
        h, w = arr.shape
        arr = arr[:h // step * step, :w // step * step].reshape(h // step, step, w // step, step).mean((1, 3)).astype(arr.dtype)
    resp = Response(arr.tobytes(), mimetype="application/octet-stream")
    resp.headers["X-Width"] = str(arr.shape[1])
    resp.headers["X-Height"] = str(arr.shape[0])
    resp.headers["X-Dtype"] = "f4" if arr.dtype.kind == "f" else "u2"
    resp.headers["Access-Control-Expose-Headers"] = "X-Width, X-Height, X-Dtype"
    resp.headers["Cache-Control"] = "max-age=3600"
    return resp


_trace_jobs: dict[str, dict] = {}


@app.route("/api/videotrace/<key>", methods=["POST", "GET"])
def api_videotrace(key):
    """Per-frame mean |frame/ref - 1| inside an ROI: a video 'activity' signal to plot next to the
    acoustic emission. POST starts a job {ref, roi:[x0,y0,x1,y1], step}; GET polls it."""
    if request.method == "GET":
        j = _trace_jobs.get(key)
        return jsonify(j or {"state": "none"})
    v = video(key)
    if not v:
        abort(404)
    cfg = request.get_json(force=True)
    ref_n = int(cfg.get("ref", v.first))
    x0, y0, x1, y1 = [int(x) for x in cfg.get("roi", [0, 0, v.meta["width"], v.meta["height"]])]
    step = max(1, int(cfg.get("step", 1)))
    ref_k = max(1, min(200, int(cfg.get("refavg", 1))))
    dk = dark(key) if cfg.get("dark") else None
    dk = dk[y0:y1, x0:x1] if dk is not None else 0.0
    job = {"state": "running", "done": 0, "total": 0, "t": [], "y": [], "roi": [x0, y0, x1, y1], "ref": ref_n}
    _trace_jobs[key] = job

    def run():
        try:
            rns = [m for m in range(ref_n, ref_n + ref_k) if v.first <= m <= v.last] or [ref_n]
            ref = np.mean([v.read_frame(m)[y0:y1, x0:x1].astype(np.float32) for m in rns], 0) - dk
            ref[ref < 1] = 1
            frames = list(range(v.first, v.last + 1, step))
            job["total"] = len(frames)
            fps = v.meta["fps"]
            for i, n in enumerate(frames):
                if job.get("cancel"):
                    job["state"] = "cancelled"
                    return
                fr = v.read_frame(n)[y0:y1, x0:x1].astype(np.float32) - dk
                job["t"].append(n / fps * 1e3)
                job["y"].append(float(np.mean(np.abs(fr / ref - 1.0))))
                job["done"] = i + 1
            job["state"] = "done"
        except Exception as e:
            job["state"] = "error"
            job["error"] = str(e)

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"state": "running"})


# ------------------------------------------------------------------------------- auto-detection
CACHE_DIR = batch.CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)
_det_jobs: dict[str, dict] = {}
_sum_lock = threading.Lock()


def _cfg(cfg):
    c = {**batch.DEFAULT_CFG, **{k: v for k, v in (cfg or {}).items() if k in batch.DEFAULT_CFG}}
    c["step"] = max(1, int(c["step"]))
    return c


def _update_summary(summ):
    with _sum_lock:
        d = batch.load_summary()
        d[summ["key"]] = summ
        batch.save_summary(d)


def _build_job(key, cfg, laser):
    """Full detection result for the viewer from the caches (acoustic is recomputed, it is fast)."""
    s = sample(key)
    job = {"state": "done", "cfg": cfg, "laser": laser, "acoustic": None, "video": None}
    if s.get("acoustic_file"):
        ch = _acache.get(s["acoustic_file"])
        gate = detect.laser_window(ch)
        job["laser"] = gate
        if "RMS" in ch:  # some steel files only recorded the laser gate (A2)
            job["acoustic"] = detect.detect_acoustic(ch, gate, thr_V=cfg["thr_V"], min_dwell_ms=cfg["dwell_ms"])
        else:
            job["acoustic_note"] = "laser gate (A2) only, no acoustic channel A1"
    v = video(key)
    if v:
        path = batch.video_cache_path(key, v, cfg["video_thr"], cfg["step"])
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            an = json.load(f)
        lw = job["laser"] or laser
        evs = detect.video_events(an, thr_px=cfg["thr_px"], min_dwell_ms=cfg["dwell_ms"], laser=lw)
        an["events"] = evs
        an["stats"] = detect.open_stats(an, evs)
        an["speed"] = detect.scan_speed(an)
        job["video"] = an
    return job


@app.route("/api/detect/<key>", methods=["POST"])
def api_detect_start(key):
    """Start / refresh auto-detection for one sample. Body: {thr_V, thr_px, dwell_ms, video_thr, step, force}"""
    body = request.get_json(force=True) or {}
    cfg = _cfg(body)
    s = sample(key)
    laser, _assumed = batch.laser_window(s)
    job = {"state": "running", "done": 0, "total": 0, "acoustic": None, "laser": laser}
    _det_jobs[key] = job

    def run():
        try:
            summ = batch.analyse_sample(s, cfg, force=bool(body.get("force")))
            _update_summary(summ)
            full = _build_job(key, cfg, laser)
            if _det_jobs.get(key) is job:
                if full is None:
                    job.update({"state": "error", "error": "; ".join(summ["errors"]) or "no result"})
                else:
                    _det_jobs[key] = full
        except Exception as e:  # noqa
            job["state"] = "error"
            job["error"] = repr(e)

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"state": "running"})


@app.route("/api/detect/<key>", methods=["GET"])
def api_detect_get(key):
    j = _det_jobs.get(key)
    if not j:
        # results from an earlier (batch) run are picked up from the cache
        summ = batch.load_summary().get(key)
        if summ:
            laser, _ = batch.laser_window(sample(key))
            j = _build_job(key, _cfg(summ.get("cfg")), laser)
            if j:
                _det_jobs[key] = j
        if not j:
            return jsonify({"state": "none"})
    if j["state"] != "done":
        return jsonify({k: j[k] for k in ("state", "done", "total", "acoustic", "laser") if k in j} |
                       ({"error": j["error"]} if "error" in j else {}))
    return jsonify(j)


# --------------------------------------------------------------------------------------- batch
_batch = {"state": "idle", "done": 0, "total": 0, "current": [], "errors": [], "started": None}


@app.route("/api/batch", methods=["POST"])
def api_batch_start():
    if _batch["state"] == "running":
        return jsonify(_batch)
    body = request.get_json(force=True) or {}
    rescan()
    cfg = _cfg(body)
    force = bool(body.get("force"))
    workers = max(1, min(6, int(body.get("workers", 3))))
    todo = [dict(s) for s in _index["samples"].values() if s.get("video_folder") or s.get("acoustic_file")]
    _batch.update({"state": "running", "done": 0, "total": len(todo), "current": [], "errors": [],
                   "started": time.time(), "cfg": cfg})
    _det_jobs.clear()

    def run():
        from concurrent.futures import ProcessPoolExecutor, as_completed
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(batch.analyse_sample, s, cfg, force): s["key"] for s in todo}
                _batch["current"] = list(futs.values())
                for fu in as_completed(futs):
                    k = futs[fu]
                    try:
                        summ = fu.result()
                        _update_summary(summ)
                        for e in summ["errors"]:
                            _batch["errors"].append(f"{k}: {e}")
                    except Exception as e:  # noqa
                        _batch["errors"].append(f"{k}: {e!r}")
                    _batch["done"] += 1
                    _batch["current"] = [x for x in _batch["current"] if x != k]
            _batch["state"] = "done"
        except Exception as e:  # noqa
            _batch["state"] = "error"
            _batch["errors"].append(repr(e))
        _batch["elapsed_s"] = round(time.time() - _batch["started"], 1)

    threading.Thread(target=run, daemon=True).start()
    return jsonify(_batch)


@app.route("/api/batch", methods=["GET"])
def api_batch_get():
    return jsonify(_batch)


@app.route("/api/hysteresis")
def api_hysteresis():
    if not _index["samples"]:
        rescan()
    basis = request.args.get("basis", "auto")
    merge_ms = float(request.args.get("merge_ms", 1.0))
    summ = batch.load_summary()
    rows = []
    for key, s in _index["samples"].items():
        laser = None
        sm = summ.get(key)
        if sm:
            laser = sm.get("laser")
        pi = data.process_info(s.get("params"), tuple(laser) if laser and not (sm or {}).get("laser_assumed") else None)
        manual = load_ann(key)["events"] if os.path.exists(ann_path(key)) else []
        rows.append(hysteresis.sample_row(s, sm, pi, manual, basis=basis, merge_ms=merge_ms))
    return jsonify(hysteresis.evaluate(rows, by_wall=request.args.get("group") == "wall"))


@app.route("/api/detect/<key>/video.csv")
def api_detect_csv(key):
    j = _det_jobs.get(key)
    if not j or not j.get("video"):
        abort(404, "run auto-detect first")
    an = j["video"]
    um = float(request.args.get("um_per_px", 0) or 0)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["frame", "t_ms", "depth_px", "depth_um", "front_angle_deg", "rear_angle_deg", "tip_x_px",
                "tip_y_px", "area_px"])
    for n, t, d, a, ra, pf in zip(an["frames"], an["t"], an["depth"], an["angle"], an["rear_angle"], an["per_frame"]):
        w.writerow([n, t, d, f"{d * um:.2f}" if um else "", "" if a is None else a, "" if ra is None else ra,
                    pf["tip_x"] if pf else "", pf["tip_y"] if pf else "", pf["area"] if pf else ""])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={key}_keyhole_metrics.csv"})


@app.route("/api/annotations/<key>", methods=["GET", "PUT"])
def api_ann(key):
    sample(key)
    if request.method == "PUT":
        a = request.get_json(force=True)
        save_ann(key, a)
        return jsonify({"ok": True})
    return jsonify(load_ann(key))


@app.route("/api/export.csv")
def api_export():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["sample", "material", "wall", "side", "line", "event", "source", "time_ms",
                "frame", "x_along_line_mm", "power_at_event_W", "power_profile", "velocity_mm_s",
                "velocity_source", "laser_on_ms", "laser_off_ms", "sync_offset_ms", "note"])
    for fn in sorted(os.listdir(ANN_DIR)):
        if not fn.endswith(".json"):
            continue
        a = json.load(open(os.path.join(ANN_DIR, fn), encoding="utf-8"))
        key = a.get("key", fn[:-5])
        s = _index["samples"].get(key, {})
        pi = process(key) if key in _index["samples"] else None
        prof = (pi or {}).get("profile") or {}
        lms = (pi or {}).get("laser_ms") or ["", ""]
        for e in sorted(a.get("events", []), key=lambda e: e["t_ms"]):
            x, pw = data.power_at(pi, e["t_ms"])
            w.writerow([s.get("id", key), s.get("material", ""), s.get("wall", ""), s.get("side", ""),
                        s.get("line", ""), e["type"], e["source"], f"{e['t_ms']:.4f}", e.get("frame", ""),
                        "" if x is None else x, "" if pw is None else pw, prof.get("text", ""),
                        (pi or {}).get("velocity_mm_s", ""), (pi or {}).get("v_source", ""), lms[0], lms[1],
                        a.get("sync_offset_ms", 0), e.get("note", "")])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=keyhole_events.csv"})


if __name__ == "__main__":
    import webbrowser
    rescan()
    port = int(os.environ.get("PORT", 5019))
    if not os.environ.get("NO_BROWSER"):
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False)
