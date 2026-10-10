import json
import math
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

BOTTLE_CLASS = 39  # COCO class ID for "bottle"
CONF_THRESHOLD = 0.4

CROSSHAIR_SIZE = 20  # half-length of each arm, px
CROSSHAIR_COLOR = (0, 255, 0)  # BGR
CROSSHAIR_THICKNESS = 2
CROSSHAIR_Y_OFFSET = 20  # px up from bottom edge, so the lower arm isn't clipped

TARGET_LINE_COLOR = (0, 255, 255)  # BGR
TARGET_LINE_THICKNESS = 2
INTERCEPT_LINE_COLOR = (255, 0, 255)  # BGR
INTERCEPT_LINE_THICKNESS = 2

KNOWN_DISTANCE_CM = 25.0  # D: bottle-to-camera distance when pressing C
BOTTLE_REAL_HEIGHT_CM = 17.5  # H: real bottle height
CALIBRATION_FILE = "calibration.json"
EDGE_MARGIN = 2  # px; a box this close to the top/bottom edge is probably clipped

DEFAULT_SPEED_MPS = 5.0  # v: assumed interceptor speed
SPEED_STEP_MPS = 1.0
MIN_SPEED_MPS = 1.0
# cv2.waitKeyEx codes (Windows, Linux/GTK)
KEYS_SPEED_UP = {2490368, 2555904, 65362, 65363}  # Up, Right
KEYS_SPEED_DOWN = {2621440, 2424832, 65364, 65361}  # Down, Left

# Intercept mode (toggle with M)
VELOCITY_WINDOW_S = 0.5  # fit velocity over this much recent history
MIN_TRACK_SAMPLES = 5  # samples needed before a velocity is trusted
TRACK_LOST_FRAMES = 30  # frames without the target before the lock is released (matches ByteTrack's track_buffer)
MIN_TARGET_SPEED_MPS = 0.05  # slower than this is treated as stationary (filters jitter)

WARN_COLOR = (0, 0, 255)  # BGR

HUD_MARGIN = 10  # px from the bottom-left corner
HUD_PADDING = 10  # px inside the panel
HUD_ALPHA = 0.6  # panel background opacity
HUD_TEXT_SCALE = 0.55
HUD_LABEL_COLOR = (200, 200, 200)  # BGR
HUD_VALUE_COLOR = (255, 255, 255)  # BGR
HUD_LOCK_COLOR = (0, 255, 0)  # BGR
FPS_SMOOTHING = 0.9  # 0 = raw per-frame FPS, closer to 1 = smoother
BOX_HEIGHT_SMOOTHING = 0.7  # EMA factor for box height (steadies distance); 0 = raw
CENTER_SMOOTHING = 0.5  # EMA factor for box center; kept light so fast motion doesn't lag; 0 = raw


def draw_crosshair(img):
    """Draw a fixed crosshair at the bottom center of img and return its (x, y) origin."""
    h, w = img.shape[:2]
    x, y = w // 2, h - 1 - CROSSHAIR_Y_OFFSET
    cv2.line(img, (x - CROSSHAIR_SIZE, y), (x + CROSSHAIR_SIZE, y), CROSSHAIR_COLOR, CROSSHAIR_THICKNESS)
    cv2.line(img, (x, y - CROSSHAIR_SIZE), (x, y + CROSSHAIR_SIZE), CROSSHAIR_COLOR, CROSSHAIR_THICKNESS)
    cv2.circle(img, (x, y), 3, CROSSHAIR_COLOR, -1)
    return x, y


def draw_hud(img, status, status_color, rows):
    """Draw a semi-transparent HUD panel in the bottom-left corner.

    rows is a list of (label, value) strings; status is the panel heading.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (_, th), baseline = cv2.getTextSize("Ag", font, HUD_TEXT_SCALE, 1)
    row_h = th + baseline + 6
    label_w = max(cv2.getTextSize(label, font, HUD_TEXT_SCALE, 1)[0][0] for label, _ in rows)
    value_w = max(cv2.getTextSize(value, font, HUD_TEXT_SCALE, 1)[0][0] for _, value in rows)
    status_w = cv2.getTextSize(status, font, HUD_TEXT_SCALE, 2)[0][0]
    panel_w = max(label_w + 20 + value_w, status_w) + 2 * HUD_PADDING
    panel_h = row_h * (len(rows) + 1) + 2 * HUD_PADDING

    img_h = img.shape[0]
    x0, y0 = HUD_MARGIN, max(0, img_h - HUD_MARGIN - panel_h)
    x1, y1 = x0 + panel_w, y0 + panel_h

    # Darken the panel area toward black instead of covering it
    img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * (1 - HUD_ALPHA)).astype(img.dtype)
    cv2.rectangle(img, (x0, y0), (x1, y1), status_color, 1)

    tx, ty = x0 + HUD_PADDING, y0 + HUD_PADDING + th
    cv2.putText(img, status, (tx, ty), font, HUD_TEXT_SCALE, status_color, 2, cv2.LINE_AA)
    for label, value in rows:
        ty += row_h
        cv2.putText(img, label, (tx, ty), font, HUD_TEXT_SCALE, HUD_LABEL_COLOR, 1, cv2.LINE_AA)
        cv2.putText(img, value, (tx + label_w + 20, ty), font, HUD_TEXT_SCALE, HUD_VALUE_COLOR, 1, cv2.LINE_AA)


def all_bottles(result):
    """Return a list of ((x1, y1, x2, y2), confidence, track_id) for every tracked bottle.

    track_id is None when the tracker has not assigned IDs yet.
    """
    boxes = result.boxes
    ids = boxes.id.tolist() if boxes.id is not None else [None] * len(boxes)
    return [
        (boxes.xyxy[i].tolist(), float(boxes.conf[i]), None if ids[i] is None else int(ids[i]))
        for i in range(len(boxes))
    ]


def ema(prev, new, alpha):
    """Exponential moving average: alpha * prev + (1 - alpha) * new. The first sample (prev is None) is taken as is."""
    return new if prev is None else alpha * prev + (1 - alpha) * new


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def select_bottle(detections, locked_id, last_center):
    """Pick the bottle to follow.

    With no locked track ID, take the most confident one. Otherwise take the detection
    carrying that ID. If that ID is gone (ByteTrack often assigns a new one when the
    bottle moves fast), fall back to the detection nearest the last known position.
    """
    if not detections:
        return None
    if locked_id is None:
        return max(detections, key=lambda d: d[1])
    for d in detections:
        if d[2] == locked_id:
            return d
    lx, ly = last_center
    return min(detections, key=lambda d: math.hypot(box_center(d[0])[0] - lx, box_center(d[0])[1] - ly))


def focal_length(box_height):
    return (box_height * KNOWN_DISTANCE_CM) / BOTTLE_REAL_HEIGHT_CM


def distance_cm(f_px, box_height):
    return (f_px * BOTTLE_REAL_HEIGHT_CM) / box_height


def bearing_deg(offset_px, f_px):
    """Angle off the optical axis for a pixel offset from the image center: theta = atan(x / f)."""
    return math.degrees(math.atan(offset_px / f_px))


def pixel_to_camera(cx, cy, box_height, f_px, frame_w, frame_h):
    """Back-project a box center to a camera-frame position in metres (x right, y up, z forward)."""
    z = distance_cm(f_px, box_height) / 100
    x = (cx - frame_w / 2) * z / f_px
    y = (frame_h / 2 - cy) * z / f_px
    return np.array([x, y, z])


def camera_to_pixel(p, f_px, frame_w, frame_h):
    """Project a camera-frame point (metres) back to pixel coordinates, or None if behind the camera."""
    x, y, z = p
    if z <= 0.01:
        return None
    return frame_w / 2 + f_px * x / z, frame_h / 2 - f_px * y / z


def solve_intercept(p, v_target, interceptor_speed):
    """Earliest time t > 0 at which an interceptor leaving the camera at interceptor_speed
    can meet a target at p moving at constant v_target: |p + v_target * t| = interceptor_speed * t.

    Returns t in seconds, or None if no intercept is possible.
    """
    a = v_target @ v_target - interceptor_speed**2
    b = 2 * (p @ v_target)
    c = p @ p
    if abs(a) < 1e-9:
        return -c / b if b < 0 else None
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sq = math.sqrt(disc)
    roots = [t for t in ((-b - sq) / (2 * a), (-b + sq) / (2 * a)) if t > 0]
    return min(roots) if roots else None


class TargetTracker:
    """Remembers which tracked bottle ID is being followed, plus a short history of its
    3D positions to which it fits a constant-velocity line."""

    def __init__(self):
        self.history = deque()  # (timestamp, position)
        self.target_id = None  # ByteTrack ID of the bottle we are following
        self.last_center = None  # last raw box center in pixels, for re-locking after an ID change
        self.smooth_height = None  # EMA of box height in pixels
        self.smooth_center = None  # EMA of box center (x, y) in pixels
        self.missed = 0

    def reset(self):
        self.history.clear()
        self.target_id = None
        self.last_center = None
        self.clear_smoothing()
        self.missed = 0

    def clear_smoothing(self):
        self.smooth_height = None
        self.smooth_center = None

    def lock(self, track_id, center):
        """Follow track_id (if the tracker gave one), remember where it is, and clear the miss counter."""
        if track_id is not None:
            if self.target_id is not None and track_id != self.target_id:
                self.history.clear()  # ID changed, so old positions may not be the same bottle
                self.clear_smoothing()
            self.target_id = track_id
        self.last_center = center
        self.missed = 0

    def smooth(self, height, center):
        """Fold a raw box height and center into their EMAs and return the smoothed (height, center)."""
        self.smooth_height = ema(self.smooth_height, height, BOX_HEIGHT_SMOOTHING)
        prev_x, prev_y = self.smooth_center or (None, None)
        self.smooth_center = (
            ema(prev_x, center[0], CENTER_SMOOTHING),
            ema(prev_y, center[1], CENTER_SMOOTHING),
        )
        return self.smooth_height, self.smooth_center

    def update(self, t, position):
        self.history.append((t, position))
        while self.history and t - self.history[0][0] > VELOCITY_WINDOW_S:
            self.history.popleft()

    def mark_missed(self):
        self.missed += 1
        self.clear_smoothing()  # a stale average would drag the box when the bottle comes back
        if self.missed > TRACK_LOST_FRAMES:
            self.reset()

    def estimate(self, t_now):
        """Return (position, velocity) at t_now from a least-squares line fit, or None if too few
        samples or the newest one is older than VELOCITY_WINDOW_S (stale)."""
        if len(self.history) < MIN_TRACK_SAMPLES:
            return None
        if t_now - self.history[-1][0] > VELOCITY_WINDOW_S:
            return None
        ts = np.array([s[0] for s in self.history]) - t_now
        ps = np.array([s[1] for s in self.history])
        velocity, position = np.polyfit(ts, ps, 1)  # slope, intercept (at t_now) for each axis
        if np.linalg.norm(velocity) < MIN_TARGET_SPEED_MPS:
            velocity = np.zeros(3)
        return position, velocity


def load_calibration():
    try:
        with open(CALIBRATION_FILE) as f:
            return json.load(f)["focal_length_px"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None


def save_calibration(box_height, f_px, clipped):
    data = {
        "focal_length_px": f_px,
        "known_distance_cm": KNOWN_DISTANCE_CM,
        "real_height_cm": BOTTLE_REAL_HEIGHT_CM,
        "box_height_px": box_height,
        "clipped": clipped,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(data, f, indent=2)


def main():
    model = YOLO("yolo26n.pt")
    saved_f = load_calibration()
    speed = DEFAULT_SPEED_MPS
    intercept_mode = False
    tracker = TargetTracker()
    fps = None
    last_time = time.perf_counter()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (device 0).")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam.")
                break

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[BOTTLE_CLASS],
                conf=CONF_THRESHOLD,
                verbose=False,
            )
            annotated = results[0].plot()
            origin = draw_crosshair(annotated)
            frame_h, frame_w = frame.shape[:2]

            now = time.perf_counter()
            inst_fps = 1.0 / max(now - last_time, 1e-6)
            fps = ema(fps, inst_fps, FPS_SMOOTHING)
            last_time = now

            # Compute everything first, then draw
            detections = all_bottles(results[0])
            detection = select_bottle(detections, tracker.target_id, tracker.last_center)
            box_height = clipped = f_px = conf = None
            d_cm = t_s = theta_h = theta_v = None
            target_speed = aim_h = aim_v = None
            intercept_px = None
            if detection is not None:
                (x1, y1, x2, y2), conf, track_id = detection
                raw_center = box_center((x1, y1, x2, y2))
                tracker.lock(track_id, raw_center)
                # Smoothed height and center feed distance, bearing and the 3D position;
                # the edge-clipping check below still uses the raw box.
                box_height, (cx, cy) = tracker.smooth(y2 - y1, raw_center)
                center = (int(cx), int(cy))
                cv2.line(annotated, origin, center, TARGET_LINE_COLOR, TARGET_LINE_THICKNESS)
                cv2.circle(annotated, center, 4, TARGET_LINE_COLOR, -1)
                f_px = focal_length(box_height)
                clipped = y1 <= EDGE_MARGIN or y2 >= frame_h - 1 - EDGE_MARGIN

                if saved_f is not None and box_height:
                    # Distance and time to hit, using the saved focal length
                    d_cm = distance_cm(saved_f, box_height)
                    t_s = (d_cm / 100) / speed
                    # Bearing angles, measured from the image center (optical axis); +x right, +y up
                    theta_h = bearing_deg(cx - frame_w / 2, saved_f)
                    theta_v = bearing_deg(frame_h / 2 - cy, saved_f)

                    if intercept_mode:
                        position = pixel_to_camera(cx, cy, box_height, saved_f, frame_w, frame_h)
                        tracker.update(now, position)
            else:
                tracker.mark_missed()

            # Intercept prediction: replaces the direct time to hit with the lead solution
            intercept_ok = False
            if intercept_mode and saved_f is not None:
                estimate = tracker.estimate(now)
                if estimate is not None:
                    position, velocity = estimate
                    target_speed = float(np.linalg.norm(velocity))
                    t_hit = solve_intercept(position, velocity, speed)
                    t_s = None
                    if t_hit is not None:
                        intercept_ok = True
                        t_s = t_hit
                        aim = position + velocity * t_hit
                        aim_h = math.degrees(math.atan2(aim[0], aim[2]))
                        aim_v = math.degrees(math.atan2(aim[1], aim[2]))
                        intercept_px = camera_to_pixel(aim, saved_f, frame_w, frame_h)
                        if intercept_px is not None:
                            ip = (int(intercept_px[0]), int(intercept_px[1]))
                            cv2.line(annotated, origin, ip, INTERCEPT_LINE_COLOR, INTERCEPT_LINE_THICKNESS)
                            cv2.drawMarker(annotated, ip, INTERCEPT_LINE_COLOR, cv2.MARKER_TILTED_CROSS, 16, 2)

            # HUD panel
            if saved_f is None:
                status, status_color = ("TARGET - NOT CALIBRATED" if detection else "NO TARGET"), WARN_COLOR
            elif intercept_mode:
                if target_speed is None:
                    status, status_color = ("TRACKING..." if detection else "NO TARGET"), WARN_COLOR
                elif intercept_ok:
                    status, status_color = "INTERCEPT SOLUTION", HUD_LOCK_COLOR
                else:
                    status, status_color = "NO INTERCEPT - TOO FAST", WARN_COLOR
            elif detection is None:
                status, status_color = "NO TARGET", WARN_COLOR
            else:
                status, status_color = "TARGET LOCKED", HUD_LOCK_COLOR
            na = "--"
            rows = [
                ("MODE", "INTERCEPT (M)" if intercept_mode else "DIRECT (M)"),
                ("DIST", f"{d_cm:.1f} cm" if d_cm is not None else na),
                ("TIME TO HIT", f"{t_s:.3f} s" if t_s is not None else na),
                ("SPEED", f"{speed:g} m/s"),
                ("BEARING H", f"{theta_h:+.1f} deg" if theta_h is not None else na),
                ("BEARING V", f"{theta_v:+.1f} deg" if theta_v is not None else na),
            ]
            if intercept_mode:
                rows += [
                    ("TGT SPEED", f"{target_speed:.2f} m/s" if target_speed is not None else na),
                    ("AIM H", f"{aim_h:+.1f} deg" if aim_h is not None else na),
                    ("AIM V", f"{aim_v:+.1f} deg" if aim_v is not None else na),
                ]
            rows += [
                ("CONF", f"{conf:.2f}" if conf is not None else na),
                ("FPS", f"{fps:.1f}"),
            ]
            draw_hud(annotated, status, status_color, rows)

            cv2.imshow("Webcam", annotated)

            key = cv2.waitKeyEx(1)
            if key == ord("q"):
                break
            if key in (ord("m"), ord("M")):
                intercept_mode = not intercept_mode
                tracker.reset()
            elif key in KEYS_SPEED_UP:
                speed += SPEED_STEP_MPS
            elif key in KEYS_SPEED_DOWN:
                speed = max(MIN_SPEED_MPS, speed - SPEED_STEP_MPS)
            elif key in (ord("c"), ord("C")):
                if f_px is None:
                    print("No bottle detected, calibration skipped.")
                else:
                    save_calibration(box_height, f_px, clipped)
                    saved_f = f_px
                    tracker.reset()  # old positions were computed with the previous f
                    print(
                        f"Calibrated: f = ({box_height:.1f} x {KNOWN_DISTANCE_CM:g}) / "
                        f"{BOTTLE_REAL_HEIGHT_CM:g} = {f_px:.1f} px -> {CALIBRATION_FILE}"
                    )
                    if clipped:
                        print("Warning: box touched the frame edge, so h (and f) is likely too small.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
