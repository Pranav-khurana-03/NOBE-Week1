# Webcam Bottle Tracker

Live bottle detection and targeting from a webcam, built with YOLO26 nano and OpenCV. The script detects a bottle in each frame and estimates its distance, bearing and time to hit. In intercept mode it also tracks the bottle's motion and aims at the predicted intercept point.

Everything lives in a single script, `webcamTest.py`.

## Contents

- [Setup](#setup)
- [Running](#running)
- [Controls](#controls)
- [What's on screen](#whats-on-screen)
- [How it works](#how-it-works)
  - [Detection](#1-detection)
  - [Crosshair origin](#2-crosshair-origin)
  - [Calibration](#3-calibration-focal-length)
  - [Distance](#4-distance)
  - [Time to hit](#5-time-to-hit)
  - [Bearing angles](#6-bearing-angles)
  - [Intercept mode](#7-intercept-mode)
- [Configuration](#configuration)
- [Calibration results](#calibration-results)
- [Limitations and known issues](#limitations-and-known-issues)
- [Project files](#project-files)
- [Development history](#development-history)

---

## Setup

The project uses a Python virtual environment in `venv/` (Python 3.14.7).

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt`:

```
ultralytics
opencv-python
numpy
lap
```

`lap` is required by ByteTrack. The installed versions are ultralytics 8.4.171, opencv-python 5.0.0, numpy 2.5.3, lap 0.5.13 and torch 2.14.1. PyTorch is the **CPU-only** build, so inference runs on the CPU.

The YOLO26 nano weights (`yolo26n.pt`, about 5.3 MB) download automatically on first run.

## Running

```powershell
.\venv\Scripts\python.exe webcamTest.py
```

A window titled **Webcam** opens with the live feed. The OpenCV window must have focus for key presses to register.

## Controls

| Key | Action |
|---|---|
| `q` | Quit |
| `c` / `C` | Calibrate: compute and save the focal length from the current frame |
| `m` / `M` | Switch between **Direct** and **Intercept** mode |
| Up / Right arrow | Interceptor speed +1 m/s |
| Down / Left arrow | Interceptor speed −1 m/s (minimum 1 m/s) |

## What's on screen

- **Bottle box:** YOLO's labeled bounding box with its confidence score.
- **Green crosshair:** the fixed origin at the bottom center of the frame.
- **Yellow line:** from the origin to the center of the tracked bottle's box.
- **Magenta line and ✕ (intercept mode only):** from the origin to the predicted intercept point.
- **HUD panel (bottom-left):** a semi-transparent readout. The border color matches the status: green means locked or solved, red means a warning.

| HUD row | Meaning |
|---|---|
| Status | `TARGET LOCKED`, `NO TARGET`, `TARGET - NOT CALIBRATED`, or in intercept mode `TRACKING...`, `INTERCEPT SOLUTION`, `NO INTERCEPT - TOO FAST` |
| MODE | `DIRECT` or `INTERCEPT` |
| DIST | Distance to the bottle (cm) |
| TIME TO HIT | Direct: `d / v`. Intercept: the solved intercept time |
| SPEED | Assumed interceptor speed `v` (m/s) |
| BEARING H / V | Angle to the bottle's current position (degrees, + = right / up) |
| TGT SPEED | *Intercept mode:* the bottle's estimated speed (m/s) |
| AIM H / V | *Intercept mode:* angle to the predicted intercept point (degrees) |
| CONF | YOLO confidence of the tracked bottle |
| FPS | Smoothed frames per second for the whole loop |

Values that can't be computed yet, because there's no detection or no calibration, show `--`.

---

## How it works

### 1. Detection

Each frame goes through YOLO26 nano, filtered to the COCO **bottle** class (ID `39` in YOLO's 0–79 numbering) with confidence ≥ 0.35:

```python
results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[BOTTLE_CLASS], conf=CONF_THRESHOLD, verbose=False)
```

In direct mode the most confident bottle is used. In intercept mode the script follows one bottle across frames (see [Intercept mode](#7-intercept-mode)).

### 2. Crosshair origin

A fixed "+" sits at the bottom center of the frame at `(w/2, h − 1 − CROSSHAIR_Y_OFFSET)`. It's computed from the frame size, so it works at any resolution. It's the visual reference point that the target and intercept lines are drawn from.

### 3. Calibration (focal length)

Distance from a single camera needs the camera's focal length in pixels. Hold the bottle upright at a known distance `D` and press **C**. The script then computes:

```
f = (h × D) / H
```

| Symbol | Meaning |
|---|---|
| `h` | Box height in pixels |
| `D` | Known distance (`KNOWN_DISTANCE_CM`, 25 cm) |
| `H` | Real bottle height (`BOTTLE_REAL_HEIGHT_CM`, 17.5 cm) |

The result is saved to `calibration.json` along with its inputs and a timestamp. It's loaded automatically on startup. Calibration uses the current frame only.

If the box touches the top or bottom edge of the frame, the bottle is probably cut off and `h` is too small. In that case the console prints a warning, and the file records `"clipped": true`.

### 4. Distance

With `f` known, the pinhole camera model is inverted:

```
d = (f × H) / h
```

### 5. Time to hit

```
t = d / v
```

`d` is converted to metres, and `v` is the interceptor speed set with the arrow keys (default 5 m/s).

### 6. Bearing angles

```
θ = atan(x / f)
```

`x` is the pixel offset from the **image center**. The image center is the optical axis, so this is where the formula is valid. The bottom-center crosshair is not the reference here.

- Horizontal: `x = cx − w/2` (+ = right)
- Vertical: `y = h/2 − cy` (+ = up; the sign is flipped because image y increases downward)

### 7. Intercept mode

Press **M** to enable it. It runs in four steps.

**a) 3D position.** Each detection is back-projected to camera coordinates in metres (x right, y up, z forward):

```
z = (f × H) / h
x = (cx − w/2) × z / f
y = (h/2 − cy) × z / f
```

**b) Frame-to-frame tracking.** `model.track(persist=True)` runs ByteTrack, which gives each bottle a persistent ID. The script locks onto the ID of the first (most confident) bottle and keeps following it, so it doesn't jump between bottles. The lock is released after `TRACK_LOST_FRAMES` (30) frames without that ID (matching ByteTrack's `track_buffer`).

**c) Velocity.** A least-squares straight line is fitted to the last `VELOCITY_WINDOW_S` (0.5 s) of positions, and needs at least `MIN_TRACK_SAMPLES` (5) samples. The slope is the velocity, and the fitted value at the current time is a smoothed position. Speeds below `MIN_TARGET_SPEED_MPS` (0.05 m/s) are treated as zero, so a stationary bottle's jitter doesn't create a false lead.

**d) Intercept solution.** The interceptor is assumed to launch from the camera at speed `v` in a straight line. The target is assumed to keep moving at constant velocity. The two meet when:

```
|P + V·t| = v·t
```

Squaring gives a quadratic in `t`:

```
(V·V − v²)·t² + 2(P·V)·t + P·P = 0
```

The smallest positive root is the intercept time. If there isn't one, the target is moving away faster than the interceptor can fly, and the HUD shows `NO INTERCEPT - TOO FAST`.

The aim point `P + V·t` is projected back to pixels (`u = w/2 + f·x/z`, `v = h/2 − f·y/z`), and the magenta line is drawn to it. `AIM H/V` are `atan2(x, z)` and `atan2(y, z)` of the aim point.

Switching modes and recalibrating both reset the track.

---

## Configuration

All settings are constants at the top of `webcamTest.py`.

| Constant | Default | Purpose |
|---|---|---|
| `BOTTLE_CLASS` | `39` | COCO class ID for bottle |
| `CONF_THRESHOLD` | `0.35` | Minimum detection confidence |
| `CROSSHAIR_SIZE` | `20` | Crosshair arm half-length (px) |
| `CROSSHAIR_COLOR` | `(0, 255, 0)` | Crosshair color (BGR) |
| `CROSSHAIR_THICKNESS` | `2` | Crosshair line width |
| `CROSSHAIR_Y_OFFSET` | `20` | Distance of the crosshair above the bottom edge (px) |
| `TARGET_LINE_COLOR` | `(0, 255, 255)` | Origin → bottle line (yellow) |
| `TARGET_LINE_THICKNESS` | `2` | |
| `INTERCEPT_LINE_COLOR` | `(255, 0, 255)` | Origin → intercept line (magenta) |
| `INTERCEPT_LINE_THICKNESS` | `2` | |
| `KNOWN_DISTANCE_CM` | `25.0` | `D`, the calibration distance |
| `BOTTLE_REAL_HEIGHT_CM` | `17.5` | `H`, the real bottle height |
| `CALIBRATION_FILE` | `"calibration.json"` | Where the focal length is saved |
| `EDGE_MARGIN` | `2` | Distance from the frame edge at which a box counts as clipped (px) |
| `DEFAULT_SPEED_MPS` | `5.0` | Starting interceptor speed |
| `SPEED_STEP_MPS` | `1.0` | Arrow-key increment |
| `MIN_SPEED_MPS` | `1.0` | Lowest allowed speed (prevents division by zero) |
| `VELOCITY_WINDOW_S` | `0.5` | History length for the velocity fit |
| `MIN_TRACK_SAMPLES` | `5` | Samples needed before velocity is trusted |
| `TRACK_LOST_FRAMES` | `30` | Missed frames before the ID lock is released |
| `MIN_TARGET_SPEED_MPS` | `0.05` | Speed below which the target is treated as stationary |
| `HUD_*` | | HUD position, padding, opacity, text size and colors |
| `FPS_SMOOTHING` | `0.9` | FPS smoothing factor (0 = raw, closer to 1 = smoother) |
| `BOX_HEIGHT_SMOOTHING` | `0.7` | EMA factor for box height, which steadies the distance estimate (0 = raw) |
| `CENTER_SMOOTHING` | `0.5` | EMA factor for box center, kept light so fast motion doesn't lag (0 = raw) |
| `STATS_WINDOW_S` | `5.0` | Seconds between refreshes of the HUD std-dev readout (`STD H` and `STD C`, shown as raw > EMA in px) |

## Calibration results

**First calibration.** `H` = 19 cm, `D` = 25 cm.

| h (px) | f (px) |
|---|---|
| 284.4 | 374.2 |
| 284.0 | 373.7 |
| 283.5 | 373.1 |
| 283.6 | 373.2 |

Frame-to-frame noise was about 1 px in `f` (about 0.3%).

**Second calibration (current).** The bottle height was corrected to `H` = 17.5 cm, `D` = 25 cm.

| h (px) | f (px) |
|---|---|
| 277.8 | 396.8 |
| 276.4 | 394.9 |
| 303.2 | 433.2 |
| 302.8 | 432.6 ← saved |

The presses fall into two groups about 9% apart. Detection noise within each group is small, so the gap most likely comes from the bottle moving about 2 cm, or its box fitting differently, between presses. The saved value is **f = 432.6 px**.

To check it, place the bottle at a measured distance such as 40 cm and confirm that DIST reads close to that. If it reads about 43–44 cm, recalibrate with the bottle carefully at 25 cm.

## Limitations and known issues

- **CPU-only inference:** PyTorch is the CPU build, so FPS is limited. A CUDA build of torch would speed it up on an NVIDIA GPU.
- **Calibration accuracy:** errors in measuring `D` matter far more than detection noise. 1 cm at 25 cm is a 4% error in `f`, and that error carries into distance, time to hit and the angles.
- **Close-range clipping:** at 25 cm a 17.5 cm bottle fills most of the frame. If the box is cut off at the frame edge, `h` comes out too small.
- **Same-height assumption:** distance is only correct for a bottle of height `H`. A different bottle needs `BOTTLE_REAL_HEIGHT_CM` updated. The focal length stays valid.
- **Single-frame calibration:** there's no averaging over multiple frames.
- **Lens model:** the optical center is assumed to be the image center, and lens distortion is ignored, so angles near the frame edges are slightly off.
- **Intercept assumptions:** the target moves at constant velocity, and the interceptor flies straight from the camera at constant speed. Depth (z) from box height is the noisiest axis, so motion toward or away from the camera is estimated less reliably than sideways motion.
- **Small lead at close range:** at 25 cm and 5 m/s the intercept time is only about 0.05 s, so the aim point sits almost on the bottle. Lowering the speed to 1 m/s makes the lead visible.
- **Webcam disconnects:** one run stopped after about 10 minutes with the Windows Media Foundation error `-1072873822` ("video recording device invalidated"). That happens when another app takes the camera, the device sleeps, or the camera disconnects. The script exits with `Failed to read frame from webcam.` Possible fixes are automatic reconnection or the `cv2.CAP_DSHOW` backend.

## Project files

| File | Description |
|---|---|
| `webcamTest.py` | Main script |
| `requirements.txt` | Python dependencies |
| `calibration.json` | Saved focal length and calibration inputs (created by pressing C) |
| `yolo26n.pt` | YOLO26 nano weights (auto-downloaded) |
| `venv/` | Python virtual environment |

## Development history

1. **Environment.** Confirmed `venv` (Python 3.14.7) with ultralytics, OpenCV and numpy installed; torch is CPU-only.
2. **Webcam feed.** `webcamTest.py` shows a live OpenCV window that closes on `q`.
3. **Bottle detection.** Added YOLO26 nano inference on every frame, filtered to COCO class 39 at ≥ 0.5 confidence, with labeled boxes.
4. **Crosshair origin.** Added a fixed crosshair at the bottom center, with its settings as constants.
5. **Calibration mode.** Pressing `C` computes `f = (h × D) / H` from a single frame and saves it to `calibration.json`. The working was shown on screen so the noise could be measured.
6. **Distance and time to hit.** Added `d = (f × H) / h` and `t = d / v`, with the interceptor speed adjustable by arrow keys (default 5 m/s, 1 m/s steps).
7. **Bottle height update.** Changed `H` from 19 cm to 17.5 cm and recalibrated (f = 432.6 px).
8. **Target line and bearings.** Added the origin-to-box-center line and horizontal and vertical bearings `θ = atan(x / f)`, measured from the image center.
9. **HUD panel.** Added a bottom-left panel showing distance, time to hit, speed, bearings, confidence and FPS.
10. **HUD cleanup.** Removed the on-screen calculation text, leaving only the HUD.
11. **Intercept mode.** Added 3D tracking across frames, least-squares velocity estimation, a closed-form intercept solution, and a second (magenta) aim line to the predicted intercept point, toggled with `M`.
