# Keyhole Viewer

Synchronized browser for high-speed X-ray videos (Photron, `.tif` + `.cihx`) and acoustic emission
recordings (Elsys TranAX `.tpc5`) of laser line scans, with marking and automatic detection of keyhole
opening / collapse, keyhole depth and front-wall angle, and an evaluation of the laser power at which the
keyhole forms and collapses.

## Start
Put this folder inside the data folder (it scans its parent directory for videos, acoustics and the
parameter sheet `LinkWallNumberParameters.xlsx`). Double-click `start_viewer.bat` (installs
`requirements.txt`, then runs `python app.py`) → opens http://127.0.0.1:5019

## Data discovery
The parent folder is scanned recursively; press **Rescan** after more data is copied.
- Videos: any folder with `.cihx` files named `W<wall>[S<side>]L<line>…` (flat/dark-field folders are skipped).
- Acoustics: any `W…L….tpc5` file.
- Material from the path: contains `Titan` → Ti64, `Steel`/`316` → 316L.
- Video and acoustic are paired by material + sample ID.
- Process parameters (power profile, velocity) come from the parameter sheet (one sheet per material).
  Two powers `a;b` = linear ramp over the 4 mm line; three powers = 0 / 2 / 4 mm unless the *Profile* column
  gives the positions in mm.
- Samples can be left out of the index with `EXCLUDE_SAMPLES` in `data.py`.

## Time base
- Acoustic: t = (sample − triggerSample) / fs. Channel A1 = microphone, A2 = laser gate (>2.5 V = on).
  Files without A1 still provide the laser timing.
- Video: frame n (relative to the camera trigger) at t = n / fps + sync offset (default 0 ms, settable per sample).
- "RMS" = 50 µs moving RMS of A1 after removing a 2 ms baseline.

## Controls
- ←/→ ±1 frame, Shift+←/→ ±10, Space play/pause
- Plot: click = move cursor (sub-frame resolution), drag = zoom, wheel = zoom, right-drag = pan, double-click = laser window
- Q / W = keyhole opens / collapses seen in **video**; A / S = seen in **acoustic**
- **Processed ⇄ Raw switch** (or key R): *Processed* = the selected correction (÷ reference frame by default,
  Δ previous frame, or no correction) with the min/max contrast fields; *Raw* = untreated pixel values with a
  linear brightness/contrast stretch per frame — "sample" (stretches the sample population) or
  "full range" (0.3–99.7 % of all pixels). Flip and keyhole overlay apply in both.
- Display: flip ↕ (default on: sample at the bottom, keyhole pointing down) / flip ↔.
- The bottom-right corner of the video shows frame number, time, commanded power and position along the line.
- Shift+drag on the image = ROI, then **Video trace** = per-frame mean |frame/ref − 1| in the ROI, overlaid on the plot

Annotations are stored locally in `annotations/<sample>.json`; **Export CSV** collects all of them.

## Auto-detect (keyhole opening / collapse, depth, front-wall angle)
Press **Run auto-detect** in a sample. Acoustic results come instantly; the video analysis takes ~20–40 s
and is cached in `annotations/cache/`. Results appear as dotted suggestion lines; **Accept** (or **Accept all**)
copies them into the marked events with the note "auto".

- **Acoustic**: keyhole open while the A1 RMS (0.2 ms smoothed) is above *RMS thr* (Schmitt trigger ×/÷ 1.15).
  Openings are taken at the threshold crossing. After a collapse the acoustic signal rings down exponentially
  (`DECAY_TAU_MS` in `detect.py`), so the threshold is crossed too late; the collapse time is therefore the
  onset of that decay (least-squares fit of a level followed by an exponential decay), but not earlier than the
  last time the RMS was still at `OPEN_FRACTION` of its open level. The plain threshold crossing is kept in the
  event as `t_threshold_ms`. A collapse at or after laser off is reported at the laser-off time.
- **Video**: every frame is divided by a pre-laser and a post-laser reference (median of frames);
  pixels brighter than both by *pixel thr* are vapour. The keyhole is the bright component that starts at the
  sample surface and reaches deepest. The surface is found per column in the pre-laser reference (yellow dashed
  line), so sloped or uneven tops are handled.
  - depth = local surface → tip [px]; shown in µm with the *pixel size* field.
  - front-wall angle = line fit to the sub-pixel front edge between 15 % and 90 % of the depth,
    measured from the vertical; positive = tip trails behind the opening. Scan direction is found from
    the tip trajectory.
  - open/collapse = depth above *depth thr* (with 70 % hysteresis); runs shorter than *min dwell* are ignored.
    A collapse is labelled "keyhole leaves field of view" if the keyhole tip was within 50 px of the image edge.
  - Samples without a laser-gate recording assume the laser window from the sheet and use the last frames of
    the recording as post-laser reference.
- Right axis of the plot: keyhole depth / front-wall angle / video trace. **Per-frame depth/angle CSV**
  exports frame, t, depth, front & rear angle, tip position.

## Hysteresis view (left menu → "Hysteresis")
- **Rescan + auto-detect all samples** runs the acoustic + video detection on every sample in 3 worker
  processes (video results are cached, "recompute" forces a re-run). Results are collected in
  `annotations/cache/summary.json`; opening a sample afterwards shows them immediately.
- Every opening/collapse is mapped onto the commanded power at that moment and classified:
  `on` (opening on a rising ramp → P_on), `off` (collapse on a falling ramp → P_off), `initial`, `end`/`fov`
  (still open at laser off / leaving the image → upper bound on P_off), `none` (never formed → lower bound on P_on),
  `*_flat` (at constant power), `on_falling`/`off_rising` (against the ramp, not used).
  Open/closed periods shorter than the flicker filter (default 1 ms) are ignored.
- Per material × speed (× wall, default) and sensor: mean P_on, P_off, ΔP ± s.e., ramp rate and constant-power
  outcomes.
- Use "marked (where present)" to evaluate your own accepted/marked events instead of the raw auto-detection.
