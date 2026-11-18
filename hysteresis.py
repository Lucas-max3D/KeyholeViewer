"""Power hysteresis of keyhole formation / collapse.

For every line the detected keyhole openings and collapses (acoustic and video separately) are mapped
onto the commanded power profile (sheet power vs. position, position = v * (t - t_laser_on)).

Classification of each transition:
  on        opening while the power is rising           -> formation power  P_on
  off       collapse while the power is falling         -> collapse power   P_off
  on_flat / off_flat   transition at constant power (delayed formation / spontaneous collapse)
  on_falling / off_rising   transition against the ramp (instability, reported but not used)
  initial   opening within `init_ms` of laser on       -> P_on <= P(start) (upper bound)
  end       still open when the laser switched off     -> P_off <= P(end) (upper bound)
  fov       keyhole leaves the video field of view     -> P_off <= P there (upper bound)
  none      no keyhole on the whole line               -> P_on > max P of the line (lower bound)

Flicker is removed first: open periods shorter than `merge_ms` are dropped and gaps shorter than
`merge_ms` are bridged, so only regime changes remain.

Hysteresis = keyhole forms at a higher power (on a rising ramp) than the power at which it collapses
(on a falling ramp): dP = mean(P_on) - mean(P_off) > 0 beyond the scatter. Caveat: a finite response
time tau shifts P_on up by rate*tau and P_off down by rate*tau, so a purely dynamic lag also produces
dP ~ 2*rate*tau; comparing groups with different ramp rates (W/ms) separates the two.
"""
from __future__ import annotations

import numpy as np

import data

INIT_MS = 1.0


def _slope(prof, x):
    """dP/dx [W/mm] of the piecewise-linear profile at x (right-sided at breakpoints)."""
    xs, ps = prof["x_mm"], prof["P_W"]
    for i in range(1, len(xs)):
        if x < xs[i] or i == len(xs) - 1:
            dx = xs[i] - xs[i - 1]
            return (ps[i] - ps[i - 1]) / dx if dx else 0.0
    return 0.0


def _intervals(events, t_on, t_off, merge_ms):
    """Events -> list of open intervals [t_open, t_close, close_note, open_note]."""
    ev = sorted(events, key=lambda e: e["t_ms"])
    iv, cur = [], None
    for e in ev:
        if e["type"] == "opening":
            if cur is None:
                cur = [e["t_ms"], None, "", e.get("note", "")]
        elif cur is not None:
            cur[1], cur[2] = e["t_ms"], e.get("note", "")
            iv.append(cur)
            cur = None
    if cur is not None:  # never closed -> treat as open until laser off
        cur[1], cur[2] = t_off, "at laser off"
        iv.append(cur)
    # bridge short gaps, then drop short open periods
    merged = []
    for it in iv:
        if merged and it[0] - merged[-1][1] < merge_ms:
            merged[-1][1], merged[-1][2] = it[1], it[2]
        else:
            merged.append(list(it))
    return [it for it in merged if it[1] - it[0] >= merge_ms]


def _classify(pi, iv, src):
    prof, v = pi["profile"], pi["velocity_mm_s"]
    t_on, t_off = pi["laser_ms"]
    out = []

    def at(t):
        x, p = data.power_at(pi, min(max(t, t_on), t_off))
        return x, p, _slope(prof, x or 0.0) * v / 1e3  # rate in W/ms

    for t_o, t_c, cnote, onote in iv:
        x, p, r = at(t_o)
        if t_o - t_on < INIT_MS:
            cls = "initial"
        elif abs(r) < 1e-9:
            cls = "on_flat"
        else:
            cls = "on" if r > 0 else "on_falling"
        out.append({"kind": "opening", "cls": cls, "t_ms": round(t_o, 4), "x_mm": x, "P_W": p, "rate_W_ms": round(r, 3)})
        x, p, r = at(t_c)
        note = (cnote or "").lower()
        if "laser off" in note or t_c >= t_off - 0.05 or "field of view" in note:
            base = "fov" if "field of view" in note else "end"
            # only a bound on P_off if the power was not rising there
            x2, p2, r2 = at(t_c - 0.05)
            cls = base if r2 <= 1e-9 else base + "_rising"
        elif abs(r) < 1e-9:
            cls = "off_flat"
        else:
            cls = "off" if r < 0 else "off_rising"
        out.append({"kind": "collapse", "cls": cls, "t_ms": round(t_c, 4), "x_mm": x, "P_W": p, "rate_W_ms": round(r, 3)})
    return out


def sample_row(s, summ, pi, manual, basis="auto", merge_ms=1.0):
    row = {"key": s["key"], "id": s["id"], "material": s["material"], "wall": s["wall"], "side": s["side"],
           "line": s["line"], "analysed": bool(summ), "usable": False, "sources": {}, "notes": []}
    if pi:
        row.update({"profile": pi["profile"], "v": pi["velocity_mm_s"], "laser_ms": pi.get("laser_ms"),
                    "notes": list(pi.get("notes", []))})
    if not summ:
        row["notes"].append("not analysed yet")
        return row
    if not pi or not pi.get("profile") or not pi.get("velocity_mm_s") or not pi.get("laser_ms"):
        row["notes"].append("no power profile / velocity / laser timing")
        return row
    if summ.get("laser_assumed"):
        row["laser_assumed"] = True
    row["usable"] = True
    t_on, t_off = pi["laser_ms"]
    prof = pi["profile"]
    row["ramp"] = "constant" if len(set(prof["P_W"])) == 1 else "ramp"
    for src in ("acoustic", "video"):
        auto = (summ.get(src) or {}).get("events") if summ.get(src) else None
        man = [e for e in manual if e.get("source") == src]
        if basis == "manual" and man:
            evs, used = man, "manual"
        elif auto is not None:
            evs, used = auto, "auto"
        else:
            continue
        iv = _intervals(evs, t_on, t_off, merge_ms)
        cls = _classify(pi, iv, src)
        res = {"basis": used, "transitions": cls, "keyhole": bool(iv)}
        if not iv:
            res["transitions"] = [{"kind": "none", "cls": "none", "P_W": max(prof["P_W"]), "t_ms": None,
                                   "x_mm": None, "rate_W_ms": None}]
        if src == "video":
            sp = (summ.get("video") or {}).get("speed")
            res["speed_px_ms"] = sp["px_per_ms"] if sp else None
        row["sources"][src] = res
    return row


def _stats(vals):
    a = np.array(vals, float)
    if a.size == 0:
        return None
    return {"n": int(a.size), "mean": round(float(a.mean()), 2), "sd": round(float(a.std(ddof=1)), 2) if a.size > 1 else None,
            "min": round(float(a.min()), 2), "max": round(float(a.max()), 2)}


def evaluate(rows, by_wall=False):
    groups = {}
    for r in rows:
        if not r.get("usable"):
            continue
        vn = float(round(r["v"] / 10.0) * 10) if r["v"] else None  # nominal speed (gate-inferred ~202 -> 200)
        wall = r["wall"] if by_wall else None
        g = groups.setdefault((r["material"], vn, wall or 0),
                              {"material": r["material"], "v": vn, "wall": wall, "lines": 0, "sources": {}})
        g["lines"] += 1
        for src, res in r["sources"].items():
            gs = g["sources"].setdefault(src, {"on": [], "off": [], "on_rate": [], "off_rate": [], "initial": [],
                                               "end": [], "none": [], "flat_on": [], "flat_off": [],
                                               "against": [], "line_dP": [], "const_yes": [], "const_no": []})
            ons = [t for t in res["transitions"] if t["cls"] == "on"]
            offs = [t for t in res["transitions"] if t["cls"] == "off"]
            for t in res["transitions"]:
                c = t["cls"]
                if c == "on":
                    gs["on"].append(t["P_W"]); gs["on_rate"].append(t["rate_W_ms"])
                elif c == "off":
                    gs["off"].append(t["P_W"]); gs["off_rate"].append(t["rate_W_ms"])
                elif c == "initial":
                    gs["initial"].append(t["P_W"])
                elif c in ("end", "fov"):
                    gs["end"].append(t["P_W"])
                elif c == "none":
                    gs["none"].append(t["P_W"])
                elif c == "on_flat":
                    gs["flat_on"].append(t["P_W"])
                elif c == "off_flat":
                    gs["flat_off"].append(t["P_W"])
                elif c in ("on_falling", "off_rising"):
                    gs["against"].append(t["P_W"])
            if ons and offs:
                gs["line_dP"].append({"id": r["id"], "dP": round(ons[0]["P_W"] - offs[0]["P_W"], 2),
                                      "P_on": ons[0]["P_W"], "P_off": offs[0]["P_W"]})
            if r.get("ramp") == "constant":
                (gs["const_yes"] if res["keyhole"] else gs["const_no"]).append(r["profile"]["P_W"][0])

    out_groups = []
    for (mat, v, _w), g in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0, kv[0][2])):
        for src, gs in g["sources"].items():
            son, soff = _stats(gs["on"]), _stats(gs["off"])
            verdict, why = "insufficient data", ""
            dP = se = None
            if son and soff:
                dP = round(son["mean"] - soff["mean"], 2)
                if son["n"] >= 2 and soff["n"] >= 2:
                    var = son["sd"] ** 2 / son["n"] + soff["sd"] ** 2 / soff["n"]
                    se = round(float(np.sqrt(var)), 2)
                    tol = max(2 * se, 3.0)
                    sfx = ""
                    why = f"ΔP = {dP:+.1f} W (±{se:.1f} W s.e.)"
                else:  # scatter unknown from a single transition
                    tol, sfx = 5.0, " (indicative, n=1)"
                    why = f"ΔP = {dP:+.1f} W from single transitions"
                if dP > tol:
                    verdict = "hysteresis" + sfx
                elif dP < -tol:
                    verdict = "inverse (collapse above formation power)" + sfx
                else:
                    verdict = "no significant hysteresis" + sfx
            # bounds: keyhole survives ramp-downs to P_end while rising ramps need P_on to form it
            if gs["end"] and son:
                surv_min = min(gs["end"])
                if surv_min < son["min"] - 3:
                    extra = (f"keyhole kept open down to {surv_min:g} W at laser off, below the lowest formation "
                             f"power {son['min']:g} W")
                    why = (why + "; " if why else "") + extra
                    if verdict in ("insufficient data", "no significant hysteresis") and not soff:
                        verdict = "hysteresis (from bounds)"
            # history test: a power that gives no keyhole from a cold start (constant-power line) but at which
            # an existing keyhole survives when the power is brought down to it -> bistable -> hysteresis
            bist = sorted({p for p in gs["end"] + gs["flat_off"] for q in gs["const_no"] if p <= q + 0.5})
            if bist:
                extra = (f"history dependence: keyhole survives at {', '.join(f'{p:g}' for p in bist)} W after a higher "
                         f"power, but constant {', '.join(f'{q:g}' for q in sorted(set(gs['const_no'])))} W from a cold "
                         f"start gives no keyhole")
                why = (why + "; " if why else "") + extra
                if not verdict.startswith("hysteresis"):
                    verdict = "hysteresis (history dependence)"
            rates = [abs(x) for x in gs["on_rate"] + gs["off_rate"]]
            out_groups.append({
                "material": mat, "v": v, "wall": g["wall"], "source": src, "lines": g["lines"], "bistable_P": bist,
                "P_on": son, "P_off": soff, "dP": dP, "se": se, "verdict": verdict, "why": why,
                "mean_rate_W_ms": round(float(np.mean(rates)), 3) if rates else None,
                "initial_P": gs["initial"], "survived_to_P": gs["end"], "never_formed_max_P": gs["none"],
                "flat_on_P": gs["flat_on"], "flat_off_P": gs["flat_off"], "against_ramp_P": gs["against"],
                "line_dP": gs["line_dP"], "const_keyhole": sorted(gs["const_yes"]), "const_no_keyhole": sorted(gs["const_no"]),
                "raw_on": gs["on"], "raw_off": gs["off"],
            })
    n_an = sum(1 for r in rows if r["analysed"])
    return {"groups": out_groups, "rows": sorted(rows, key=lambda r: (r["material"], r["wall"], r["side"] or 0, r["line"])),
            "n_samples": len(rows), "n_analysed": n_an, "n_usable": sum(1 for r in rows if r.get("usable"))}
