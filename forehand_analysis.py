#!/usr/bin/env python3
"""
Forehand Analysis: Pose + Face Blur + Slow Motion + Biomechanical Metrics
─────────────────────────────────────────────────────────────────────────
v1 features:
  - Joint angle overlays (elbow, shoulder, knee)
  - Hip-shoulder separation angle
  - Swing phase detection (Ready → Backswing → Loading → Acceleration → Follow-Through)
  - Wrist trajectory arc visualization
  - Keypoint temporal smoothing (reduces jitter)
  - Stance detection (Open / Semi-Open / Closed)
  - Per-frame CSV metrics export
  - On-screen metrics HUD panel

v2 additions:
  - Wrist speed (px/frame proxy for racket-head speed) + per-swing peak speed badge
  - Contact point classifier (Early / Ideal / Late) via arm-to-torso angle at impact;
    fires automatically at the Acceleration → Follow-Through transition
  - Kinetic chain sequencing: detects hip → shoulder → wrist firing order per swing
    and displays ms gaps between each segment; shown as a badge after Follow-Through
"""

import argparse
import csv
import os
import sys
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

SUPPORTED_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")

# YOLOv8-Pose keypoint indices (COCO 17-point skeleton)
KP = {
    "nose": 0,
    "l_shoulder": 5, "r_shoulder": 6,
    "l_elbow": 7,    "r_elbow": 8,
    "l_wrist": 9,    "r_wrist": 10,
    "l_hip": 11,     "r_hip": 12,
    "l_knee": 13,    "r_knee": 14,
    "l_ankle": 15,   "r_ankle": 16,
}

PHASE_LABELS = ["Ready", "Backswing", "Loading", "Acceleration", "Follow-Through"]

SMOOTH_WINDOW = 5
TRAIL_LENGTH  = 30
HUD_WIDTH     = 300
HUD_ALPHA     = 0.65

# Contact point: arm-to-torso angle thresholds (degrees).
# At ideal contact the dominant arm is roughly perpendicular to the torso (~90-125 deg).
CONTACT_EARLY_MAX = 80    # < 80  → Early (arm rushed ahead)
CONTACT_LATE_MIN  = 125   # > 125 → Late  (arm still lagging)

# Kinetic chain firing thresholds
HIP_FIRE_DELTA      = 3.0   # abs deg/frame change in hip orientation
SHOULDER_FIRE_DELTA = 3.0   # abs deg/frame change in shoulder orientation
WRIST_SPEED_FIRE    = 60.0  # wrist px/frame speed

# Frames to display contact / chain badges after they trigger
BADGE_FRAMES = 90


# ──────────────────────────────────────────────
# Arguments
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Forehand Analysis: Pose + Face Blur + Biomechanics"
    )
    parser.add_argument("--input", "-i")
    parser.add_argument("--input-dir")
    parser.add_argument("--output", "-o")
    parser.add_argument("--output-dir")
    parser.add_argument("--pose-model",    default="yolov8m-pose.pt")
    parser.add_argument("--face-model",    default="yolov8n-face.pt")
    parser.add_argument("--device",        default="cpu")
    parser.add_argument("--conf",          type=float, default=0.5)
    parser.add_argument("--face-conf",     type=float, default=0.4)
    parser.add_argument("--slow-factor",   type=int,   default=4)
    parser.add_argument("--no-slowmo",     action="store_true",
                        help="Skip the slow-motion replay section at the end")
    parser.add_argument("--no-blur",       action="store_true",
                        help="Skip face blurring")
    parser.add_argument("--no-pose",       action="store_true",
                        help="Skip pose detection and all pose-derived overlays")
    parser.add_argument("--smooth-window", type=int, default=SMOOTH_WINDOW,
                        help="Keypoint smoothing window (frames)")
    parser.add_argument("--trail-length",  type=int, default=TRAIL_LENGTH,
                        help="Wrist trail history length (frames)")
    parser.add_argument("--export-csv",    action="store_true",
                        help="Export per-frame metrics to CSV alongside output video")
    parser.add_argument("--dominant-hand", choices=["right", "left"], default="right",
                        help="Dominant (hitting) hand for metric calculation")
    return parser.parse_args()


def validate_args(args):
    if args.input and args.input_dir:
        sys.exit("Use either --input OR --input-dir")
    if not args.input and not args.input_dir:
        sys.exit("Provide --input or --input-dir")
    if args.input and not args.output:
        sys.exit("--output required for single mode")
    if args.input_dir and not args.output_dir:
        sys.exit("--output-dir required for batch mode")


# ──────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────
def angle_between(a, b, c):
    """Angle (degrees) at vertex b in the triangle a-b-c."""
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    ba = a - b
    bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def vector_angle_horizontal(p1, p2):
    """Angle of line p1→p2 relative to horizontal (degrees)."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return float(np.degrees(np.arctan2(-dy, dx)))


# ──────────────────────────────────────────────
# Keypoint smoother
# ──────────────────────────────────────────────
class KeypointSmoother:
    """Confidence-weighted rolling average over all 17 keypoints."""

    def __init__(self, window):
        self.history = [deque(maxlen=window) for _ in range(17)]

    def update(self, kps):
        smoothed = np.zeros_like(kps)
        for i in range(17):
            self.history[i].append(kps[i])
            arr   = np.array(self.history[i])
            confs = arr[:, 2]
            if confs.sum() > 0:
                smoothed[i, :2] = np.average(arr[:, :2], axis=0, weights=confs)
                smoothed[i, 2]  = arr[-1, 2]
            else:
                smoothed[i] = kps[i]
        return smoothed

    def reset(self):
        for d in self.history:
            d.clear()


# ──────────────────────────────────────────────
# Keypoint accessor
# ──────────────────────────────────────────────
def get_keypoint(kps, name, min_conf=0.3):
    idx    = KP[name]
    x, y, c = kps[idx]
    return (int(x), int(y)) if c >= min_conf else None


# ──────────────────────────────────────────────
# Wrist speed tracker  (NEW v2)
# ──────────────────────────────────────────────
class WristSpeedTracker:
    """
    Computes wrist speed in pixels/frame each update.
    Tracks the per-swing peak speed; resets peak when a new Backswing starts.
    """

    def __init__(self):
        self.prev_pos   = None
        self.speed      = 0.0
        self.peak_speed = 0.0
        self._prev_phase = "Ready"

    def update(self, wrist_pos, phase):
        # Reset peak at the start of each new backswing
        if phase == "Backswing" and self._prev_phase == "Ready":
            self.peak_speed = 0.0

        self._prev_phase = phase

        if wrist_pos is None or self.prev_pos is None:
            self.speed    = 0.0
            self.prev_pos = wrist_pos
            return self.speed

        dx = wrist_pos[0] - self.prev_pos[0]
        dy = wrist_pos[1] - self.prev_pos[1]
        self.speed    = float(np.sqrt(dx * dx + dy * dy))
        self.peak_speed = max(self.peak_speed, self.speed)
        self.prev_pos = wrist_pos
        return self.speed


# ──────────────────────────────────────────────
# Contact point detector  (NEW v2)
# ──────────────────────────────────────────────
class ContactPointDetector:
    """
    Fires at the Acceleration → Follow-Through transition.

    Measures the angle at the shoulder between the torso vector (hip→shoulder)
    and the arm vector (shoulder→wrist).  This is orientation-independent —
    it gives the same reading regardless of which direction the player faces.

      arm_angle < CONTACT_EARLY_MAX  → "Early"
      arm_angle > CONTACT_LATE_MIN   → "Late"
      otherwise                      → "Ideal"

    The badge persists for BADGE_FRAMES frames after detection.
    """

    BADGE_COLORS = {
        "Early": (0, 100, 255),
        "Ideal": (0, 210, 80),
        "Late":  (0, 200, 255),
    }

    def __init__(self):
        self.prev_phase      = "Ready"
        self.label           = None
        self.arm_angle       = None
        self.badge_countdown = 0

    def update(self, kps, phase, dominant):
        triggered        = (self.prev_phase == "Acceleration"
                            and phase == "Follow-Through")
        self.prev_phase  = phase

        if triggered:
            dom      = "r" if dominant == "right" else "l"
            hip      = get_keypoint(kps, f"{dom}_hip")
            shoulder = get_keypoint(kps, f"{dom}_shoulder")
            wrist    = get_keypoint(kps, f"{dom}_wrist")

            if all([hip, shoulder, wrist]):
                self.arm_angle = round(angle_between(hip, shoulder, wrist), 1)
                if self.arm_angle < CONTACT_EARLY_MAX:
                    self.label = "Early"
                elif self.arm_angle > CONTACT_LATE_MIN:
                    self.label = "Late"
                else:
                    self.label = "Ideal"
                self.badge_countdown = BADGE_FRAMES
                print(f"[INFO] Contact: {self.label}  (arm angle {self.arm_angle}°)")

        if self.badge_countdown > 0:
            self.badge_countdown -= 1
            return self.label
        return None

    def reset(self):
        self.prev_phase      = "Ready"
        self.label           = None
        self.arm_angle       = None
        self.badge_countdown = 0


# ──────────────────────────────────────────────
# Kinetic chain tracker  (NEW v2)
# ──────────────────────────────────────────────
class KineticChainTracker:
    """
    Per-swing, records the frame on which each segment first "fires":
      - Hips      : abs change in hip-line angle > HIP_FIRE_DELTA deg/frame
      - Shoulders : abs change in shoulder-line angle > SHOULDER_FIRE_DELTA deg/frame
      - Wrist     : wrist speed > WRIST_SPEED_FIRE px/frame

    After Follow-Through begins, the sequence and ms gaps are locked and
    displayed as a badge for BADGE_FRAMES frames.
    """

    SEG_COLORS = {
        "Hip":      (255, 180, 0),
        "Shoulder": (0, 220, 255),
        "Wrist":    (0, 255, 128),
    }

    def __init__(self):
        self._reset_swing()
        self.result          = None
        self.badge_countdown = 0
        self.prev_phase      = "Ready"

    def _reset_swing(self):
        self.prev_hip_angle      = None
        self.prev_shoulder_angle = None
        self.hip_fire_frame      = None
        self.shoulder_fire_frame = None
        self.wrist_fire_frame    = None
        self.active              = False

    def update(self, kps, phase, wrist_speed, frame_idx, fps):
        # Arm a new swing on Backswing entry
        if phase == "Backswing" and self.prev_phase == "Ready":
            self._reset_swing()
            self.active = True

        # Lock result when Follow-Through begins
        if phase == "Follow-Through" and self.prev_phase == "Acceleration":
            self._lock_result(frame_idx, fps)
            self.active = False

        if phase == "Ready":
            self.active = False

        self.prev_phase = phase

        if self.active:
            # Hip firing detection
            l_hip = get_keypoint(kps, "l_hip")
            r_hip = get_keypoint(kps, "r_hip")
            if l_hip and r_hip:
                hip_angle = vector_angle_horizontal(l_hip, r_hip)
                if (self.prev_hip_angle is not None
                        and self.hip_fire_frame is None
                        and abs(hip_angle - self.prev_hip_angle) > HIP_FIRE_DELTA):
                    self.hip_fire_frame = frame_idx
                self.prev_hip_angle = hip_angle

            # Shoulder firing detection
            l_sh = get_keypoint(kps, "l_shoulder")
            r_sh = get_keypoint(kps, "r_shoulder")
            if l_sh and r_sh:
                sh_angle = vector_angle_horizontal(l_sh, r_sh)
                if (self.prev_shoulder_angle is not None
                        and self.shoulder_fire_frame is None
                        and abs(sh_angle - self.prev_shoulder_angle) > SHOULDER_FIRE_DELTA):
                    self.shoulder_fire_frame = frame_idx
                self.prev_shoulder_angle = sh_angle

            # Wrist firing detection
            if wrist_speed > WRIST_SPEED_FIRE and self.wrist_fire_frame is None:
                self.wrist_fire_frame = frame_idx

        # Tick badge countdown
        if self.badge_countdown > 0:
            self.badge_countdown -= 1

        return self.result if self.badge_countdown > 0 else None

    def _lock_result(self, frame_idx, fps):
        fired = {k: v for k, v in {
            "Hip":      self.hip_fire_frame,
            "Shoulder": self.shoulder_fire_frame,
            "Wrist":    self.wrist_fire_frame,
        }.items() if v is not None}

        if len(fired) < 2:
            return   # insufficient data this swing

        ordered     = sorted(fired.items(), key=lambda x: x[1])
        ms_per_frame = 1000.0 / fps
        sequence    = []
        for i, (seg, f) in enumerate(ordered):
            gap = 0 if i == 0 else round((f - ordered[i - 1][1]) * ms_per_frame)
            sequence.append({"seg": seg, "gap_ms": gap})

        self.result          = sequence
        self.badge_countdown = BADGE_FRAMES
        print(f"[INFO] Kinetic chain: "
              f"{' → '.join(s['seg'] for s in sequence)}")


# ──────────────────────────────────────────────
# Metrics computation
# ──────────────────────────────────────────────
def compute_metrics(kps, dominant="right", frame_idx=0, fps=30.0):
    dom     = "r" if dominant == "right" else "l"
    metrics = {"frame": frame_idx, "time_s": round(frame_idx / fps, 3)}

    shoulder = get_keypoint(kps, f"{dom}_shoulder")
    elbow    = get_keypoint(kps, f"{dom}_elbow")
    wrist    = get_keypoint(kps, f"{dom}_wrist")
    hip      = get_keypoint(kps, f"{dom}_hip")
    knee     = get_keypoint(kps, f"{dom}_knee")
    ankle    = get_keypoint(kps, f"{dom}_ankle")

    metrics["elbow_angle"] = (
        round(angle_between(shoulder, elbow, wrist), 1)
        if all([shoulder, elbow, wrist]) else None)

    metrics["shoulder_elevation"] = (
        round(angle_between(hip, shoulder, elbow), 1)
        if all([hip, shoulder, elbow]) else None)

    metrics["knee_bend"] = (
        round(angle_between(hip, knee, ankle), 1)
        if all([hip, knee, ankle]) else None)

    l_sh  = get_keypoint(kps, "l_shoulder")
    r_sh  = get_keypoint(kps, "r_shoulder")
    l_hip = get_keypoint(kps, "l_hip")
    r_hip = get_keypoint(kps, "r_hip")
    metrics["hip_shoulder_sep"] = (
        round(abs(vector_angle_horizontal(l_sh, r_sh) -
                  vector_angle_horizontal(l_hip, r_hip)), 1)
        if all([l_sh, r_sh, l_hip, r_hip]) else None)

    l_ankle = get_keypoint(kps, "l_ankle")
    r_ankle = get_keypoint(kps, "r_ankle")
    if l_ankle and r_ankle:
        fa = abs(vector_angle_horizontal(l_ankle, r_ankle))
        metrics["stance"] = "Closed" if fa < 20 else "Semi-Open" if fa < 50 else "Open"
    else:
        metrics["stance"] = None

    metrics["wrist_above_shoulder_px"] = (
        int(shoulder[1] - wrist[1]) if shoulder and wrist else None)

    return metrics


# ──────────────────────────────────────────────
# Swing phase detector
# ──────────────────────────────────────────────
class SwingPhaseDetector:
    """Wrist-velocity state machine."""

    def __init__(self, dominant="right", history=10):
        self.dominant        = dominant
        self.wrist_x_history = deque(maxlen=history)
        self.phase           = "Ready"
        self.phase_frame_count = 0

    def update(self, kps):
        dom   = "r" if self.dominant == "right" else "l"
        wrist = get_keypoint(kps, f"{dom}_wrist")
        if wrist is None:
            return self.phase

        self.wrist_x_history.append(wrist[0])
        if len(self.wrist_x_history) < 3:
            return self.phase

        xs            = list(self.wrist_x_history)
        recent_delta  = xs[-1] - xs[0]
        direction     = 1 if self.dominant == "right" else -1

        if abs(recent_delta) < 8:
            new_phase = "Ready"
        elif recent_delta * direction > 25:
            new_phase = "Backswing"
        elif recent_delta * direction > 5:
            new_phase = "Loading"
        elif recent_delta * direction < -40:
            new_phase = "Follow-Through"
        elif recent_delta * direction < -10:
            new_phase = "Acceleration"
        else:
            new_phase = self.phase

        if new_phase != self.phase:
            self.phase             = new_phase
            self.phase_frame_count = 0
        else:
            self.phase_frame_count += 1

        return self.phase

    def reset(self):
        self.wrist_x_history.clear()
        self.phase             = "Ready"
        self.phase_frame_count = 0


# ──────────────────────────────────────────────
# Face blur
# ──────────────────────────────────────────────
def blur_faces(frame, face_model, conf):
    results = face_model(frame, conf=conf, verbose=False)[0]
    if results.boxes is None:
        return frame
    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(face, (51, 51), 0)
    return frame


# ──────────────────────────────────────────────
# Wrist trail
# ──────────────────────────────────────────────
def draw_wrist_trail(frame, trail):
    n = len(trail)
    for i in range(1, n):
        alpha = i / n
        color = (int(255 * (1 - alpha)), int(200 * alpha), int(255 * alpha))
        cv2.line(frame, trail[i - 1], trail[i],
                 color, max(1, int(3 * alpha)), cv2.LINE_AA)
    if trail:
        cv2.circle(frame, trail[-1], 5, (0, 255, 255), -1, cv2.LINE_AA)


# ──────────────────────────────────────────────
# Angle arc
# ──────────────────────────────────────────────
def draw_angle_arc(frame, vertex, p1, p2, angle,
                   color=(0, 255, 128), radius=30):
    if any(v is None for v in (vertex, p1, p2)) or angle is None:
        return
    a1 = np.degrees(np.arctan2(-(p1[1] - vertex[1]), p1[0] - vertex[0]))
    a2 = np.degrees(np.arctan2(-(p2[1] - vertex[1]), p2[0] - vertex[0]))
    cv2.ellipse(frame, vertex, (radius, radius), 0,
                -max(a1, a2), -min(a1, a2), color, 2, cv2.LINE_AA)
    mid_a = np.radians((a1 + a2) / 2)
    lx = int(vertex[0] + (radius + 16) * np.cos(mid_a))
    ly = int(vertex[1] - (radius + 16) * np.sin(mid_a))
    cv2.putText(frame, f"{angle:.0f}", (lx - 10, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────
# Contact point badge  (NEW v2)
# ──────────────────────────────────────────────
def draw_contact_badge(frame, label, arm_angle):
    if label is None:
        return
    colors = {"Early": (0, 100, 255), "Ideal": (0, 210, 80), "Late": (0, 200, 255)}
    color  = colors.get(label, (200, 200, 200))

    h, w   = frame.shape[:2]
    bw, bh = 210, 58
    bx, by = w - bw - 10, 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (bx + 4, by + 4), (bx + bw - 4, by + 30), color, -1)
    cv2.putText(frame, f"Contact: {label}", (bx + 8, by + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (10, 10, 10), 2, cv2.LINE_AA)
    if arm_angle is not None:
        cv2.putText(frame, f"Arm angle: {arm_angle}deg", (bx + 8, by + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 210, 210), 1, cv2.LINE_AA)


# ──────────────────────────────────────────────
# Kinetic chain badge  (NEW v2)
# ──────────────────────────────────────────────
def draw_kinetic_chain_badge(frame, sequence):
    if not sequence:
        return
    h, w   = frame.shape[:2]
    rows   = len(sequence)
    bw     = 210
    bh     = 28 + rows * 24
    bx     = w - bw - 10
    by     = 80   # sits below the contact badge

    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, "Kinetic Chain", (bx + 8, by + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 255), 1, cv2.LINE_AA)

    seg_colors = {"Hip": (255, 180, 0), "Shoulder": (0, 220, 255), "Wrist": (0, 255, 128)}
    for i, item in enumerate(sequence):
        seg    = item["seg"]
        gap_ms = item["gap_ms"]
        color  = seg_colors.get(seg, (200, 200, 200))
        prefix = "1st" if i == 0 else f"+{gap_ms}ms"
        cv2.putText(frame, f"{prefix}  {seg}", (bx + 8, by + 38 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────
# HUD panel
# ──────────────────────────────────────────────
PHASE_COLORS = {
    "Ready":          (180, 180, 180),
    "Backswing":      (255, 200, 0),
    "Loading":        (255, 140, 0),
    "Acceleration":   (0, 200, 255),
    "Follow-Through": (0, 255, 128),
}


def draw_hud(frame, metrics, phase, wrist_speed, peak_speed):
    x0, y0  = 10, 10
    panel_h = 262
    panel_w = HUD_WIDTH

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, HUD_ALPHA, frame, 1 - HUD_ALPHA, 0, frame)

    def put(text, row, color=(230, 230, 230)):
        cv2.putText(frame, text, (x0 + 8, y0 + 22 + row * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

    # Phase badge strip
    p_color = PHASE_COLORS.get(phase, (200, 200, 200))
    cv2.rectangle(frame, (x0 + 4, y0 + 4), (x0 + panel_w - 4, y0 + 28),
                  p_color, -1)
    cv2.putText(frame, f"Phase: {phase}", (x0 + 8, y0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (10, 10, 10), 2, cv2.LINE_AA)

    def fmt(val, unit="deg"):
        return f"{val}{unit}" if val is not None else "--"

    put(f"Elbow angle:    {fmt(metrics.get('elbow_angle'))}", 1)
    put(f"Shoulder elev:  {fmt(metrics.get('shoulder_elevation'))}", 2)
    put(f"Hip-Shld sep:   {fmt(metrics.get('hip_shoulder_sep'))}", 3)
    put(f"Knee bend:      {fmt(metrics.get('knee_bend'))}", 4)
    put(f"Stance:         {metrics.get('stance') or '--'}", 5, (180, 220, 255))

    wrist_rel = metrics.get("wrist_above_shoulder_px")
    if wrist_rel is not None:
        lbl = "above" if wrist_rel > 0 else "below"
        put(f"Wrist {lbl} shldr: {abs(wrist_rel)}px", 6, (180, 255, 200))
    else:
        put("Wrist pos: --", 6)

    # Wrist speed rows (new)
    spd_color = (0, 255, 200) if wrist_speed > WRIST_SPEED_FIRE else (200, 200, 200)
    put(f"Wrist speed:    {wrist_speed:.1f} px/f", 7, spd_color)
    put(f"Peak speed:     {peak_speed:.1f} px/f", 8, (255, 200, 80))

    put(f"t = {metrics.get('time_s', 0):.2f}s", 9, (150, 150, 150))


# ──────────────────────────────────────────────
# Frame processing
# ──────────────────────────────────────────────
def process_frame(frame, pose_model, face_model, args,
                  smoother, phase_detector, wrist_trail,
                  speed_tracker, contact_detector, chain_tracker,
                  frame_idx, fps):

    metrics     = {"frame": frame_idx, "time_s": round(frame_idx / fps, 3)}
    phase       = "Ready"
    wrist_speed = 0.0

    if not args.no_pose:
        pose_results = pose_model(
            frame, conf=args.conf, device=args.device, verbose=False
        )[0]

        annotated = pose_results.plot(kpt_radius=4)

        if pose_results.keypoints is not None and len(pose_results.keypoints) > 0:
            raw_kps = pose_results.keypoints.data[0].cpu().numpy()
            kps     = smoother.update(raw_kps)

            metrics = compute_metrics(kps, args.dominant_hand, frame_idx, fps)
            phase   = phase_detector.update(kps)

            dom   = "r" if args.dominant_hand == "right" else "l"
            wrist = get_keypoint(kps, f"{dom}_wrist")

            # Wrist speed
            wrist_speed = speed_tracker.update(wrist, phase)

            # Contact point classification
            contact_label = contact_detector.update(kps, phase, args.dominant_hand)

            # Kinetic chain
            chain_result = chain_tracker.update(
                kps, phase, wrist_speed, frame_idx, fps)

            # Wrist trail
            if wrist:
                wrist_trail.append(wrist)
            draw_wrist_trail(annotated, list(wrist_trail))

            # Angle arcs
            shoulder = get_keypoint(kps, f"{dom}_shoulder")
            elbow    = get_keypoint(kps, f"{dom}_elbow")
            hip      = get_keypoint(kps, f"{dom}_hip")
            knee     = get_keypoint(kps, f"{dom}_knee")
            ankle    = get_keypoint(kps, f"{dom}_ankle")

            draw_angle_arc(annotated, elbow, shoulder, wrist,
                           metrics.get("elbow_angle"), color=(0, 220, 255))
            draw_angle_arc(annotated, shoulder, hip, elbow,
                           metrics.get("shoulder_elevation"), color=(0, 255, 128))
            draw_angle_arc(annotated, knee, hip, ankle,
                           metrics.get("knee_bend"), color=(255, 180, 0))

            # Top-right badges
            draw_contact_badge(annotated, contact_label, contact_detector.arm_angle)
            draw_kinetic_chain_badge(annotated, chain_result)

        # Top-left HUD
        draw_hud(annotated, metrics, phase, wrist_speed, speed_tracker.peak_speed)

    else:
        annotated = frame.copy()

    if not args.no_blur:
        annotated = blur_faces(annotated, face_model, args.face_conf)

    metrics["wrist_speed_px_f"] = round(wrist_speed, 2)
    return annotated, metrics, phase


# ──────────────────────────────────────────────
# Video processing
# ──────────────────────────────────────────────
def process_video(input_path, output_path, pose_model, face_model, args):
    print(f"[INFO] Processing: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[WARNING] Could not open {input_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    smoother         = KeypointSmoother(args.smooth_window)
    phase_detector   = SwingPhaseDetector(args.dominant_hand)
    wrist_trail      = deque(maxlen=args.trail_length)
    speed_tracker    = WristSpeedTracker()
    contact_detector = ContactPointDetector()
    chain_tracker    = KineticChainTracker()

    processed_frames = []
    all_metrics      = []
    frame_idx        = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, metrics, phase = process_frame(
            frame, pose_model, face_model, args,
            smoother, phase_detector, wrist_trail,
            speed_tracker, contact_detector, chain_tracker,
            frame_idx, fps
        )

        all_metrics.append({**metrics, "phase": phase})
        processed_frames.append(annotated)
        out.write(annotated)
        frame_idx += 1

    cap.release()

    if not args.no_slowmo:
        print("[INFO] Writing slow-motion section...")
        for f in processed_frames:
            for _ in range(args.slow_factor):
                out.write(f)
    else:
        print("[INFO] Slow-motion skipped (--no-slowmo)")

    out.release()
    print(f"[INFO] Saved: {output_path}")

    if args.export_csv and all_metrics:
        csv_path = os.path.splitext(output_path)[0] + "_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"[INFO] Metrics CSV saved: {csv_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    args = parse_args()
    validate_args(args)

    print("[INFO] Loading pose model...")
    pose_model = YOLO(args.pose_model)

    print("[INFO] Loading face model...")
    face_model = YOLO(args.face_model)

    if args.input:
        process_video(args.input, args.output, pose_model, face_model, args)

    if args.input_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        for filename in os.listdir(args.input_dir):
            if filename.lower().endswith(SUPPORTED_EXTENSIONS):
                process_video(
                    os.path.join(args.input_dir, filename),
                    os.path.join(args.output_dir, filename),
                    pose_model, face_model, args
                )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
