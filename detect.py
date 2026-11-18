"""Automatic keyhole detection.

Acoustic: the A1 RMS envelope is low in conduction mode and high in keyhole mode.
A (smoothed) RMS above `thr_V` = keyhole open.

Video: each frame is divided by a pre-laser reference AND a post-laser reference (frames after the
laser switched off). Pixels brighter than both (vapour where there was / will be metal) are
keyhole candidates; static changes such as the solidified track or pores cancel against the post
reference. The keyhole is the connected bright component that touches the sample surface and
reaches deepest. Per frame we measure:
  depth   – distance surface → keyhole tip [px]
  angle   – front-wall inclination from the vertical (beam direction) [deg], from a line fit to the
            sub-pixel front edge between 15 % and 90 % of the depth. Positive = the wall leans back,
            i.e. the tip trails behind the opening (usual keyhole shape).
  tip_x/y – tip position (raw image coordinates)
Open/collapse = depth crossing `thr_px` (with hysteresis and a minimum dwell time).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


# ------------------------------------------------------------------------------------ helpers
def _runs(state: np.ndarray):
    """(start, stop, value) of constant runs of a boolean array."""
    if state.size == 0:
        return []
    d = np.flatnonzero(np.diff(state.astype(np.int8))) + 1
    starts = np.concatenate([[0], d])
    stops = np.concatenate([d, [state.size]])
    return [(int(a), int(b), bool(state[a])) for a, b in zip(starts, stops)]


def hysteresis_state(y: np.ndarray, hi: float, lo: float, min_len: int) -> np.ndarray:
    """Schmitt trigger followed by removal of runs shorter than `min_len` samples."""
    st = np.zeros(y.size, bool)
    on = False
    for i, v in enumerate(y):
        if not on and v > hi:
            on = True
        elif on and v < lo:
            on = False
        st[i] = on
    # merge short runs into their neighbours (repeat until stable)
    for _ in range(5):
        runs = _runs(st)
        changed = False
        for a, b, val in runs:
            if b - a < min_len and 0 < a and b < st.size:
                st[a:b] = not val
                changed = True
        if not changed:
            break
    return st


def state_events(t: np.ndarray, st: np.ndarray, t_off: float | None = None):
    ev = []
    for i in np.flatnonzero(np.diff(st.astype(np.int8))):
        typ = "opening" if st[i + 1] else "collapse"
        tt = float(t[i + 1])
        note = ""
        if typ == "collapse" and t_off is not None and tt > t_off:
            tt, note = float(t_off), "at laser off"
        ev.append({"type": typ, "t_ms": round(tt, 5), "note": note})
    return ev


# ----------------------------------------------------------------------------------- acoustic
def detect_acoustic(chans: dict, laser=None, thr_V=0.17, smooth_ms=0.2, min_dwell_ms=0.3):
    rms = chans["RMS"]
    fs, trig = rms["fs"], rms["trig"]
    y = rms["y"].astype(np.float64)
    w = max(1, int(smooth_ms * 1e-3 * fs))
    ys = ndi.uniform_filter1d(y, w)
    if laser:
        t0, t1 = laser[0] - 2.0, laser[1] + 5.0
    else:
        t0, t1 = -5.0, (y.size - trig) / fs * 1e3
    i0 = max(0, int(t0 * 1e-3 * fs) + trig)
    i1 = min(y.size, int(t1 * 1e-3 * fs) + trig)
    dec = max(1, int(fs * 5e-6))  # evaluate on a 5 µs grid
    seg = ys[i0:i1:dec]
    t = (np.arange(i0, i1, dec) - trig) / fs * 1e3
    st = hysteresis_state(seg, thr_V * 1.15, thr_V / 1.15, int(min_dwell_ms * 1e-3 * fs / dec))
    if laser:  # keyhole can only exist while the laser is on
        st &= (t >= laser[0]) & (t <= laser[1] + 0.05)
    ev = state_events(t, st, laser[1] if laser else None)
    open_ms = float(st.sum() * dec / fs * 1e3)
    return {"events": ev, "thr_V": thr_V, "open_ms": round(open_ms, 3),
            "series": {"t": t[::4].round(4).tolist(), "y": seg[::4].round(4).tolist()}}


# -------------------------------------------------------------------------------------- video
def _smooth(a):
    return ndi.uniform_filter(a.astype(np.float32), 3)


def surface_row(img: np.ndarray) -> int:
    prof = ndi.uniform_filter1d(img.mean(1), 3)
    g = np.abs(np.diff(prof))
    g[:5] = 0
    g[-5:] = 0
    return int(np.argmax(g)) + 1


def surface_profile(img: np.ndarray, surf: int, win: int = 30) -> np.ndarray:
    """Top-surface row for every column (the wall top is not level: steel walls slope by 15-25 px across
    the image). Strongest vertical edge within surf +- win, column-smoothed, median-filtered along x."""
    g = np.abs(np.diff(ndi.uniform_filter(img.astype(np.float32), (3, 31)), axis=0))
    lo, hi = max(1, surf - win), min(img.shape[0] - 2, surf + win)
    sub = g[lo:hi]
    s = np.argmax(sub, axis=0) + lo + 1
    weak = sub.max(0) < 0.3 * np.median(sub.max(0))
    s[weak] = surf
    return ndi.median_filter(s, 51, mode="nearest").astype(int)


def _edge_subpix(row_r, xs_in, thr, side):
    """Sub-pixel position where row_r crosses thr just outside the mask on `side` (+1 right, -1 left)."""
    x = int(xs_in.max() if side > 0 else xs_in.min())
    xn = x + side
    if 0 <= xn < row_r.size:
        a, b = row_r[x], row_r[xn]
        if a != b:
            return x + side * float(np.clip((a - thr) / (a - b), 0, 1))
    return float(x)


SURF_TOL = 14
FOV_MARGIN = 50  # tip within this many px of the image edge -> keyhole (front) is leaving the image
VERSION = 4  # bump when the video analysis changes (invalidates the cache)


def analyze_frame(r: np.ndarray, surf: int, thr: float, y_off: int = 0, surf_x=None):
    """r = min(frame/pre, frame/post) - 1 for the analysis band. Returns dict or None.
    surf_x: optional per-column surface row (absolute); the surface of a component is then taken
    locally (median over its columns) instead of the single global row `surf`."""
    m = r > thr
    m = ndi.binary_opening(m, np.ones((3, 1), bool))
    # bridge small vertical gaps (the keyhole neck is often slightly less bright)
    m = ndi.binary_closing(m, np.ones((7, 1), bool))
    lab, nl = ndi.label(m)
    if nl == 0:
        return None
    best = None
    for i, sl in enumerate(ndi.find_objects(lab)):
        top, bot = sl[0].start, sl[0].stop - 1
        s_c = (int(round(float(np.median(surf_x[sl[1]])))) if surf_x is not None else surf) - y_off
        # must start near the surface (within SURF_TOL px)
        if not (top - SURF_TOL <= s_c <= bot + SURF_TOL):
            continue
        reach = max(s_c - top, bot - s_c)
        if best is None or reach > best[1]:
            best = (i + 1, reach, sl, s_c)
    if best is None:
        return None
    k, _, sl, s_loc = best
    sub = lab[sl] == k
    ys, xs = np.nonzero(sub)
    ys = ys + sl[0].start
    xs = xs + sl[1].start
    # material side = direction in which the component extends furthest from the surface
    up = s_loc - ys.min()
    down = ys.max() - s_loc
    sgn = -1 if up >= down else 1
    tip_y = ys.min() if sgn < 0 else ys.max()
    depth = abs(tip_y - s_loc)
    tip_x = float(xs[ys == tip_y].mean())
    out = {"depth": int(depth), "tip_x": round(tip_x, 2), "tip_y": int(tip_y + y_off),
           "x": round(float(xs.mean()), 2), "area": int(sub.sum()), "dir_mat": sgn,
           "surf_y": int(s_loc + y_off)}
    if depth >= 6:
        z0, z1 = 0.15 * depth, 0.9 * depth
        rows = [y for y in range(min(s_loc, tip_y), max(s_loc, tip_y) + 1) if z0 <= abs(y - s_loc) <= z1]
        zs, xl, xr = [], [], []
        for y in rows:
            xin = xs[ys == y]
            if xin.size == 0:
                continue
            zs.append(abs(y - s_loc))
            xr.append(_edge_subpix(r[y], xin, thr, +1))
            xl.append(_edge_subpix(r[y], xin, thr, -1))
        if len(zs) >= 4:
            zs = np.array(zs, float)
            for name, xe in (("R", np.array(xr)), ("L", np.array(xl))):
                b, a = np.polyfit(zs, xe, 1)  # x = a + b z
                out["fit" + name] = [round(float(a), 3), round(float(b), 4)]
            out["zfit"] = [round(float(zs.min()), 1), round(float(zs.max()), 1)]
    return out


def analyze_video(video, laser=None, thr=0.07, band=220, step=1, progress=None, cancel=None,
                  margin_ms=(1.5, 4.0)):
    """Run the per-frame keyhole analysis. `laser` = (t_on, t_off) in ms (from acoustics, or assumed from
    the sheet). Only frames from t_on - margin[0] to t_off + margin[1] are analysed."""
    fps = video.meta["fps"]
    first, last = video.first, video.last
    a_first, a_last = first, last
    if laser:
        a_first = max(first, int(np.floor((laser[0] - margin_ms[0]) * 1e-3 * fps)))
        a_last = min(last, int(np.ceil((laser[1] + margin_ms[1]) * 1e-3 * fps)))
    # pre reference: up to 20 frames before the laser (or before frame 0)
    t_on = laser[0] if laser else 0.0
    pre_end = min(last, int(np.floor(t_on * 1e-3 * fps)) - 5)
    pre_frames = list(range(max(first, pre_end - 40), pre_end, 2)) or [first]
    pre = np.median([_smooth(video.read_frame(n)) for n in pre_frames], 0)
    # post reference: the last frames, if they are well after the laser switched off
    post = None
    # without a laser gate (no acoustics) assume the recording outlasts the laser
    if (laser is None and last / fps * 1e3 > 20) or (laser and last / fps * 1e3 > laser[1] + 3.0):
        post_frames = list(range(max(first, last - 30), last + 1, 2))
        post = np.median([_smooth(video.read_frame(n)) for n in post_frames], 0)
    pre = np.maximum(pre, 1)
    surf = surface_row(pre)
    surf_x = surface_profile(pre, surf)
    h = pre.shape[0]
    y0, y1 = max(0, surf - band), min(h, surf + band)
    pre_b = pre[y0:y1]
    post_b = np.maximum(post[y0:y1], 1) if post is not None else None

    frames = list(range(a_first, a_last + 1, step))
    res = []
    for i, n in enumerate(frames):
        if cancel and cancel():
            return None
        # smoothing needs 1 row of context on each side; read only the band
        r0, r1 = max(0, y0 - 1), min(h, y1 + 1)
        f = _smooth(video.read_rows(n, r0, r1))[y0 - r0:y0 - r0 + (y1 - y0)]
        r = f / pre_b
        if post_b is not None:
            r = np.minimum(r, f / post_b)
        r -= 1.0
        a = analyze_frame(r, surf, thr, y0, surf_x)
        res.append(a)
        if progress and i % 20 == 0:
            progress(i + 1, len(frames))

    # scanning direction from tip x trajectory
    tt = np.array([n / fps * 1e3 for n in frames])
    xs = np.array([a["tip_x"] if a else np.nan for a in res])
    dd = np.array([a["depth"] if a else 0 for a in res], float)
    good = ~np.isnan(xs) & (dd >= 10)
    scan_dir = 1
    if good.sum() > 10:
        scan_dir = 1 if np.polyfit(tt[good], xs[good], 1)[0] >= 0 else -1
    front = "fitR" if scan_dir > 0 else "fitL"
    rear = "fitL" if scan_dir > 0 else "fitR"
    angle = np.full(len(res), np.nan)
    rear_angle = np.full(len(res), np.nan)
    for i, a in enumerate(res):
        if a and a.get(front):
            angle[i] = np.degrees(np.arctan(-a[front][1] * scan_dir))
        if a and a.get(rear):
            rear_angle[i] = np.degrees(np.arctan(-a[rear][1] * scan_dir))
    if progress:
        progress(len(frames), len(frames))
    return {
        "frames": frames, "t": tt.round(4).tolist(), "depth": dd.astype(int).tolist(),
        "angle": [None if np.isnan(v) else round(float(v), 2) for v in angle],
        "rear_angle": [None if np.isnan(v) else round(float(v), 2) for v in rear_angle],
        "per_frame": res, "surface_row": surf, "surface_profile": surf_x[::8].tolist(), "scan_dir": scan_dir, "front": front,
        "post_ref": post is not None, "thr": thr, "fps": fps, "width": int(pre.shape[1]), "step": step, "pre_frames": [pre_frames[0], pre_frames[-1]],
    }


def video_events(an: dict, thr_px=12, min_dwell_ms=0.3, laser=None):
    t = np.array(an["t"])
    d = ndi.median_filter(np.array(an["depth"], float), 5)
    if t.size < 2:
        return []
    dt = float(np.median(np.diff(t)))
    st = hysteresis_state(d, thr_px, thr_px * 0.7, max(1, int(round(min_dwell_ms / dt))))
    if laser:
        st &= (t >= laser[0] - 0.1)
    ev = state_events(t, st, laser[1] if laser else None)
    width = an.get("width", 1024)
    xs = np.array([a["tip_x"] if a else np.nan for a in an["per_frame"]], float)
    for e in ev:
        e["frame"] = int(round(e["t_ms"] * an.get("fps", 40000) / 1000))
        if e["type"] == "collapse" and not e["note"]:
            i = int(np.searchsorted(t, e["t_ms"]))
            prev = xs[max(0, i - 15):i]
            prev = prev[~np.isnan(prev)]
            if prev.size and (prev[-1] > width - FOV_MARGIN or prev[-1] < FOV_MARGIN):
                e["note"] = "keyhole leaves field of view"
    return ev


def open_stats(an: dict, events: list):
    t = np.array(an["t"])
    d = np.array(an["depth"], float)
    ang = np.array([np.nan if v is None else v for v in an["angle"]], float)
    opens = [e["t_ms"] for e in events if e["type"] == "opening"]
    closes = [e["t_ms"] for e in events if e["type"] == "collapse"]
    out = []
    for k, t0 in enumerate(opens):
        t1 = next((c for c in closes if c > t0), t[-1])
        m = (t >= t0) & (t < t1)
        if m.sum() == 0:
            continue
        out.append({"t0": t0, "t1": float(t1), "depth_mean": round(float(np.mean(d[m])), 2),
                    "depth_max": int(np.max(d[m])),
                    "angle_mean": round(float(np.nanmean(ang[m])), 2) if np.any(~np.isnan(ang[m])) else None,
                    "angle_std": round(float(np.nanstd(ang[m])), 2) if np.any(~np.isnan(ang[m])) else None})
    return out


def laser_window(chans: dict, level=2.5):
    a2 = chans.get("A2")
    if a2 is None:
        return None
    on = np.flatnonzero(a2["y"] > level)
    if on.size == 0:
        return None
    f = lambda i: (i - a2["trig"]) / a2["fs"] * 1e3
    return (round(float(f(on[0])), 5), round(float(f(on[-1])), 5))



def scan_speed(an: dict, min_depth=12):
    """Keyhole travel speed in px/ms from a robust line fit of the tip x position vs time."""
    t = np.array(an["t"])
    d = np.array(an["depth"], float)
    x = np.array([p["tip_x"] if p else np.nan for p in an["per_frame"]], float)
    w = an.get("width", 1024)
    m = (d >= min_depth) & ~np.isnan(x) & (x > 20) & (x < w - 20)
    if m.sum() < 10:
        return None
    tt, xx = t[m], x[m]
    p = np.polyfit(tt, xx, 1)
    keep = np.ones(tt.size, bool)
    for _ in range(3):
        r = np.abs(xx - np.polyval(p, tt))
        keep = r < 3 * np.median(r) + 2
        p = np.polyfit(tt[keep], xx[keep], 1)
    return {"px_per_ms": round(float(abs(p[0])), 3), "n": int(keep.sum()),
            "t_range": [round(float(tt[0]), 3), round(float(tt[-1]), 3)]}
