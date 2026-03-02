# 🎾 Forehand Analysis Tool

A computer-vision pipeline that processes tennis forehand footage and overlays biomechanical metrics directly onto the video. Built on [YOLOv8-Pose](https://github.com/ultralytics/ultralytics) for skeleton detection and OpenCV for rendering.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [All CLI Flags](#all-cli-flags)
4. [What You See on Screen](#what-you-see-on-screen)
   - [HUD Panel (top-left)](#hud-panel-top-left)
   - [Contact Point Badge (top-right)](#contact-point-badge-top-right)
   - [Kinetic Chain Badge (top-right, below contact)](#kinetic-chain-badge)
   - [Joint Angle Arcs](#joint-angle-arcs)
   - [Wrist Trail](#wrist-trail)
5. [Metric Reference & How to Interpret Each Number](#metric-reference--how-to-interpret-each-number)
   - [Elbow Angle](#1-elbow-angle)
   - [Shoulder Elevation](#2-shoulder-elevation)
   - [Hip–Shoulder Separation](#3-hipshoulder-separation)
   - [Knee Bend](#4-knee-bend)
   - [Stance](#5-stance)
   - [Wrist Position Relative to Shoulder](#6-wrist-position-relative-to-shoulder)
   - [Wrist Speed & Peak Speed](#7-wrist-speed--peak-speed)
   - [Contact Point (Early / Ideal / Late)](#8-contact-point-early--ideal--late)
   - [Kinetic Chain Sequence](#9-kinetic-chain-sequence)
6. [Swing Phases Explained](#swing-phases-explained)
7. [CSV Export & Offline Analysis](#csv-export--offline-analysis)
8. [Common Fault Patterns & What to Fix](#common-fault-patterns--what-to-fix)
9. [Filming Tips for Best Results](#filming-tips-for-best-results)
10. [Tuning the Thresholds](#tuning-the-thresholds)
11. [Known Limitations](#known-limitations)

---

## Installation

```bash
pip install ultralytics opencv-python numpy
```

The first run will automatically download the required model weights:
- `yolov8m-pose.pt` — medium pose model (default, good balance of speed and accuracy)
- `yolov8n-face.pt` — nano face detection model for blurring

To use a GPU:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Then pass `--device cuda` when running the script.

---

## Quick Start

**Single video:**
```bash
python forehand_analysis.py -i my_forehand.mp4 -o output.mp4
```

**Batch process a folder:**
```bash
python forehand_analysis.py --input-dir ./clips --output-dir ./processed
```

**Left-handed player, export CSV, skip slow-mo:**
```bash
python forehand_analysis.py -i clip.mp4 -o out.mp4 \
  --dominant-hand left \
  --export-csv \
  --no-slowmo
```

**Fast preview (no face blur, no slow-mo, CPU-friendly):**
```bash
python forehand_analysis.py -i clip.mp4 -o out.mp4 \
  --no-blur \
  --no-slowmo \
  --pose-model yolov8n-pose.pt
```

---

## All CLI Flags

| Flag | Default | Description |
|---|---|---|
| `-i / --input` | — | Path to a single input video |
| `--input-dir` | — | Path to folder of videos (batch mode) |
| `-o / --output` | — | Output video path (single mode) |
| `--output-dir` | — | Output folder (batch mode) |
| `--dominant-hand` | `right` | Hitting hand — `right` or `left`. Affects all arm/contact metrics |
| `--pose-model` | `yolov8m-pose.pt` | YOLOv8 pose checkpoint. Use `yolov8n-pose.pt` for speed or `yolov8l-pose.pt` for accuracy |
| `--face-model` | `yolov8n-face.pt` | Face detection model for blurring |
| `--device` | `cpu` | `cpu`, `cuda`, or `mps` (Apple Silicon) |
| `--conf` | `0.5` | Pose detection confidence threshold (0–1) |
| `--face-conf` | `0.4` | Face detection confidence threshold (0–1) |
| `--slow-factor` | `4` | Frame duplication multiplier for slow-motion replay |
| `--smooth-window` | `5` | Keypoint smoothing window in frames. Higher = smoother angles, more lag |
| `--trail-length` | `30` | How many frames of wrist trail history to draw |
| `--export-csv` | off | Write a `_metrics.csv` file alongside the output video |
| `--no-slowmo` | off | Skip the slow-motion replay section at the end |
| `--no-blur` | off | Skip face blurring |
| `--no-pose` | off | Skip all pose detection and overlays (passthrough video) |

---

## What You See on Screen

### HUD Panel (top-left)

A semi-transparent dark panel showing real-time metrics every frame. The top strip is color-coded by the current swing phase.

```
┌─────────────────────────────────┐
│  Phase: Acceleration            │  ← color-coded phase strip
│  Elbow angle:    142deg         │
│  Shoulder elev:  88deg          │
│  Hip-Shld sep:   41deg          │
│  Knee bend:      158deg         │
│  Stance:         Semi-Open      │
│  Wrist above shldr: 12px        │
│  Wrist speed:    74.3 px/f      │  ← green when firing
│  Peak speed:     89.1 px/f      │
│  t = 1.24s                      │
└─────────────────────────────────┘
```

---

### Contact Point Badge (top-right)

Fires once per swing at the moment of estimated ball contact (the transition from Acceleration to Follow-Through). Persists for ~3 seconds.

| Label | Color | Meaning |
|---|---|---|
| **Ideal** | Green | Arm angle 80–125° — contact in the correct zone in front of the body |
| **Early** | Orange-red | Arm angle < 80° — arm has swung too far across, contact is late spatially |
| **Late** | Yellow | Arm angle > 125° — arm is still behind the body at contact |

---

### Kinetic Chain Badge

Appears below the contact badge after each swing completes. Shows the order in which each body segment fired and the millisecond gap between them.

```
Kinetic Chain
1st   Hip          ← fired first
+48ms Shoulder     ← 48ms later
+61ms Wrist        ← 61ms after shoulder
```

---

### Joint Angle Arcs

Drawn directly on the skeleton at three joints:

| Arc Color | Joint | Measurement |
|---|---|---|
| Cyan | Elbow | Shoulder–Elbow–Wrist angle |
| Green | Shoulder | Hip–Shoulder–Elbow angle (elevation) |
| Orange/Yellow | Knee | Hip–Knee–Ankle angle (bend) |

The degree value floats next to each arc, updating every frame.

---

### Wrist Trail

A fading arc following the dominant wrist across the last 30 frames (configurable with `--trail-length`). Color transitions from purple (old) to cyan (recent), with a bright dot at the current position. The shape of the arc reveals the swing path — high-to-low, flat, or low-to-high.

---

## Metric Reference & How to Interpret Each Number

### 1. Elbow Angle

**What it is:** The angle at the elbow joint between the upper arm and forearm, measured as the shoulder–elbow–wrist angle.

**Scale:** 0° = fully folded, 180° = fully straight.

| Range | Interpretation |
|---|---|
| **100–130°** at contact | Ideal for topspin forehand — enough bend to load, enough extension for power |
| **< 90°** at contact | Arm is too tucked. Often means the swing is arm-only with no body rotation |
| **> 160°** at contact | Arm is locked out too early. Reduces control, increases elbow stress |
| **Stays flat throughout** | No acceleration through the arm — check if kinetic chain is firing |

**What to do if it's wrong:** If the arm is too straight throughout the swing, work on "low-to-high" swing path drills where the player consciously bends the elbow in the loading phase. If it stays too bent at contact, the player is probably late — see Contact Point badge.

---

### 2. Shoulder Elevation

**What it is:** The angle at the shoulder joint between the torso (hip→shoulder line) and the upper arm (shoulder→elbow line). Measures how high the arm is lifted relative to the body.

**Scale:** 0° = arm hanging straight down, 180° = arm straight up.

| Range | Interpretation |
|---|---|
| **70–100°** during Acceleration | Arm is driving forward at an efficient angle |
| **< 50°** at contact | Arm is too low — likely hitting with a flat or downward swing |
| **> 130°** at follow-through | Good high finish, wrapping around the shoulder correctly |
| **Drops below 60° before contact** | Player is "dropping the elbow," a common cause of netted balls |

**What to do if it's wrong:** Dropping elbow → work on "windshield wiper" finish drills. Arm too low throughout → focus on starting the forward swing from a higher position in the loading phase.

---

### 3. Hip–Shoulder Separation

**What it is:** The angular difference between the shoulder line and the hip line, both measured relative to horizontal. Represents how much the upper body is "wound up" relative to the lower body during the swing.

**Scale:** 0° = hips and shoulders perfectly aligned, 90° = maximum coil.

| Range | Interpretation |
|---|---|
| **35–55°** during Loading/Backswing | Good separation — enough coil to generate power via the stretch-shortening cycle |
| **< 20°** | Very little coil. Power must come entirely from the arm, which fatigues quickly and limits pace |
| **> 70°** | Extreme coil — can generate big power but risks lower-back stress and timing issues |
| **Drops rapidly during Acceleration** | Good — this is the hips "unwinding" and driving the shoulder around |
| **Stays high into Follow-Through** | Hips never rotated through — player is blocking the natural kinetic chain |

**What to do if it's wrong:** Low separation → work on unit-turn drills where the shoulders and racket turn as a single unit in the backswing. High separation that never releases → focus on hip drive cues ("belt buckle to the net").

---

### 4. Knee Bend

**What it is:** The angle at the dominant-side knee between the hip, knee, and ankle. Measures how bent the knee is during the swing.

**Scale:** 180° = completely straight leg, 90° = deep athletic squat.

| Range | Interpretation |
|---|---|
| **130–160°** during Ready and Loading | Good athletic stance — enough bend to load without over-squatting |
| **> 170°** (nearly straight) | Standing up straight. No leg loading, no weight transfer, reduced power |
| **< 120°** | Excessively deep. May indicate the player is trying to compensate for poor timing |
| **Increases (straightens) through Acceleration** | Correct — legs are "pushing up and through" into the shot, transferring energy upward |
| **Stays flat throughout** | No leg drive — arm-only swing |

**What to do if it's wrong:** Straight legs throughout → use the cue "sit into the ball" before contact. Legs don't straighten through contact → "jump through the ball" drill — exaggerate the upward push until it becomes natural.

---

### 5. Stance

**What it is:** Classifies foot positioning based on the angle of the line connecting both ankles relative to horizontal.

| Label | Ankle Line Angle | When it's appropriate |
|---|---|---|
| **Closed** | < 20° | Feet nearly parallel to the baseline. Traditional for slower balls and more time |
| **Semi-Open** | 20–50° | Front foot slightly open. The modern default — allows hip rotation while maintaining balance |
| **Open** | > 50° | Body fully facing the net. Maximizes hip rotation and recovery speed; requires good timing |

**How to use this:** Neither stance is inherently wrong — the best players use all three depending on the ball. The problem is when stance is **always the same** regardless of the situation. Use this metric to check if the player is adapting their stance to different ball positions, or if they're stuck in one pattern.

**Red flags:**
- Closed stance on wide balls → player will have to step across, reducing hip rotation
- Open stance on short low balls → harder to get down to the ball and still rotate

---

### 6. Wrist Position Relative to Shoulder

**What it is:** The vertical pixel distance between the dominant wrist and dominant shoulder. Positive means the wrist is above the shoulder, negative means below.

| Value | Interpretation |
|---|---|
| **Positive (above) during Follow-Through** | Classic high finish — indicates the swing went low-to-high correctly |
| **Near zero at contact** | Wrist at shoulder height — neutral/flat swing, appropriate for flatter shots |
| **Negative (below) at contact** | Wrist dropped below the shoulder at contact — often causes the ball to go into the net |
| **Large positive (> 50px) during Loading** | Player may be starting too high, making it harder to swing low-to-high |

**Note:** This is a pixel measurement so the absolute number depends on how far the player is from the camera. Focus on the **sign** and **relative change** across the swing, not the raw number.

---

### 7. Wrist Speed & Peak Speed

**What it is:** The pixel distance the dominant wrist travels between consecutive frames (px/frame). This is a proxy for racket-head speed — the actual racket head moves faster, but wrist speed tracks it proportionally.

**Note:** The exact px/frame number depends heavily on how far the camera is from the player and the video resolution. These numbers are most useful **comparatively** — comparing one swing to another, or one player to another filmed in the same setup.

| Reading | Interpretation |
|---|---|
| **Speed turns green in HUD** | Wrist has crossed the 60 px/f firing threshold — the arm is actively accelerating |
| **High peak speed, low average** | Good — speed is concentrated at the right moment (Acceleration phase) |
| **Consistently high speed throughout** | Player is "muscling" the shot — no relaxation phase, leads to fatigue and inconsistency |
| **Low peak speed** | Arm is decelerating early or never fully accelerating. Check kinetic chain — power may be leaking |
| **Peak speed occurs during Backswing** | Wrist is rushing on the way back, not on the way through — common in nervous or rushed players |

**What to do if peak speed is low:** First check whether the kinetic chain is sequencing correctly. If Hip and Shoulder fire but Wrist speed stays low, the issue is likely a grip problem or early deceleration. If nothing fires in the right order, the whole chain needs rebuilding from the hips.

---

### 8. Contact Point (Early / Ideal / Late)

**What it is:** At the moment the swing transitions from Acceleration to Follow-Through, the tool measures the angle at the dominant shoulder between the torso vector (hip→shoulder) and the arm vector (shoulder→wrist). This tells you **where the wrist is relative to the body** at the estimated contact moment — regardless of which way the player is facing in the frame.

| Label | Arm Angle | What's happening |
|---|---|---|
| **Early** | < 80° | The arm has swung too far in front of the body. Typically means the player contacted the ball late (behind the ideal contact zone) and is now pulling through |
| **Ideal** | 80–125° | The wrist is in front of the body but the arm hasn't crossed all the way through yet — the classic "contact out in front" position |
| **Late** | > 125° | The arm is still close to the body at estimated contact. The player is hitting with the ball too close to them — reducing power and control |

**Important nuance:** "Early" in this system means the arm has already swung through, which corresponds to being **positionally late** (the ball was too close to the body and the arm had to reach around). This terminology can be confusing — think of it as describing the arm position, not the timing relative to the ball bounce.

**What to do:**
- Consistently **Late** → The player needs to set up earlier. Use the cue "racket back before the bounce." The loading phase needs to start sooner.
- Consistently **Early** → The player is letting the ball get too close. Focus on moving into the ball earlier and contacting it farther out in front. "Meet the ball, don't let it come to you."

---

### 9. Kinetic Chain Sequence

**What it is:** Tracks the first frame during each swing when each segment begins to move aggressively, and reports the firing order and the millisecond gaps between each segment.

**How firing is detected:**
- **Hip:** The hip line angle changes more than 3°/frame
- **Shoulder:** The shoulder line angle changes more than 3°/frame
- **Wrist:** Wrist pixel speed exceeds 60 px/frame

**The ideal sequence is always: Hip → Shoulder → Wrist**

This is the kinetic chain principle — larger, slower muscles initiate, transferring energy to faster, smaller segments. Each transition adds velocity.

| Sequence | Interpretation |
|---|---|
| `Hip → Shoulder → Wrist` | Correct. Power flows from the ground up |
| `Shoulder → Hip → Wrist` | Shoulders are initiating before the hips. Upper-body dominant swing — common in beginners |
| `Wrist → Hip` or `Wrist first` | Pure arm swing. No body rotation. Severely limits power and causes elbow/shoulder overuse injuries |
| `Hip → Wrist` (no shoulder) | Hips fire but shoulder rotation is not being transferred — shoulder might be "blocking" the chain |

**Gap timing guide:**

| Gap | Interpretation |
|---|---|
| **30–80ms between each segment** | Excellent — tight, efficient energy transfer |
| **> 150ms between Hip and Shoulder** | Player rotates hips and then pauses — the energy dissipates before reaching the arm |
| **< 10ms between all segments** | Everything fires simultaneously — no sequential loading, arm-dominant shot |
| **Segments missing entirely** | That segment did not cross its threshold — it either did not fire meaningfully or was too slow to register |

**What to do:**
- No hip firing → "Hip drive" drills. Cue: "lead with the belt buckle."
- Hips fire but shoulder gap is too long → the player is not "connecting" the rotation through the torso. Shadow swing drills focusing on the shoulder following the hip immediately.
- Everything fires at once → slow the swing down dramatically and rebuild the sequence consciously, then gradually add speed.

---

## Swing Phases Explained

The phase state machine watches the dominant wrist's horizontal velocity to label each frame.

| Phase | Color | What's Happening |
|---|---|---|
| **Ready** | Gray | Player is stationary or resetting. Wrist has minimal horizontal movement |
| **Backswing** | Yellow | Wrist moving away from the net (positive x-direction for right-handed player) |
| **Loading** | Orange | Wrist still moving back but slowing — racket is at its furthest point, body coiling |
| **Acceleration** | Cyan | Wrist moving strongly toward the net — the forward swing is underway |
| **Follow-Through** | Green | Wrist moving rapidly across the body after the contact point |

**Using phases for coaching:** If the Acceleration phase is very short (only 2–3 frames of cyan), the player is not sustaining the forward swing through contact — they're decelerating early. If Loading never appears and it jumps straight from Backswing to Acceleration, the player has no pause at the back of the swing to let their body catch up.

---

## CSV Export & Offline Analysis

Use `--export-csv` to save a `_metrics.csv` file alongside the output video. Each row is one frame.

**Columns:**
```
frame, time_s, elbow_angle, shoulder_elevation, knee_bend, hip_shoulder_sep,
stance, wrist_above_shoulder_px, wrist_speed_px_f, phase
```

**Useful things to do with the CSV:**

Plot elbow angle over time to see the extension arc:
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("output_metrics.csv")
df.plot(x="time_s", y="elbow_angle")
plt.axhline(y=120, color="g", linestyle="--", label="ideal at contact")
plt.show()
```

Find the exact frame of peak wrist speed:
```python
peak_frame = df.loc[df["wrist_speed_px_f"].idxmax()]
print(peak_frame)
```

Compare hip–shoulder separation across multiple clips:
```python
for clip in clips:
    df = pd.read_csv(clip)
    loading = df[df["phase"] == "Loading"]
    print(f"{clip}: avg sep = {loading['hip_shoulder_sep'].mean():.1f}°")
```

---

## Common Fault Patterns & What to Fix

### "Arm swing" — no body rotation

**Signs in the data:**
- Kinetic chain shows `Wrist` firing first or only `Wrist` present
- Hip–shoulder separation stays below 20° throughout
- High wrist speed but low peak speed (arm working hard, little leverage)
- Knee bend stays flat (no leg loading)

**Fix:** Stop and rebuild. Start with shadow swings at 20% speed focusing only on hip turn before anything else moves. Use the kinetic chain badge to get visual confirmation the hips are firing before the wrist lights up.

---

### Late contact / ball too close to the body

**Signs in the data:**
- Contact badge shows **Late** repeatedly
- Elbow angle at Follow-Through transition is very high (> 130°)
- Wrist trail shows the arc happens mostly behind or beside the body

**Fix:** Move into the ball earlier. If the player is getting pushed back or camping on the baseline, they need to take the ball earlier. Cue: "step into the court and meet the ball."

---

### Inconsistent contact point

**Signs in the data:**
- Contact badge alternates between Early, Ideal, Late across different swings
- Wrist speed peaks at different phases each swing

**Fix:** Footwork. Inconsistent contact usually means inconsistent positioning relative to the ball. Record multiple swings and check whether the stance classification also varies — if it does, the player is reacting to poor positioning rather than executing a prepared stroke.

---

### No low-to-high swing path

**Signs in the data:**
- Wrist above shoulder is negative (wrist below shoulder) through the entire follow-through
- Shoulder elevation stays low (< 60°) throughout
- Wrist trail arc is flat or high-to-low

**Fix:** "Brush the ball" drills. Exaggerate finishing with the wrist above the head. The low-to-high path is what generates topspin, which is what keeps the ball in the court on aggressive shots.

---

### Rushed backswing / no loading

**Signs in the data:**
- No **Loading** phase detected — state jumps from Backswing directly to Acceleration
- Hip–shoulder separation at its peak is low (< 25°)
- Kinetic chain gap between Hip and Shoulder is > 150ms or Shoulder never fires

**Fix:** "Racket back early" drill. The player needs to complete the turn before the ball bounces, not after. If they are always rushing, their split-step timing is off — the issue may start before the swing itself.

---

## Filming Tips for Best Results

Getting good input footage dramatically improves detection accuracy:

- **Camera height:** Film from roughly waist height, perpendicular to the baseline. Avoid filming from behind — the torso occludes the dominant arm.
- **Distance:** Keep the full body visible with some margin. 5–10 meters from the player is usually ideal.
- **Frame rate:** 60fps or higher is strongly recommended. At 30fps the Acceleration phase may only be 3–4 frames long, making kinetic chain timing less precise.
- **Lighting:** Avoid strong backlighting (player silhouetted against bright sky). Overcast days give the most consistent detection.
- **Single player:** The model selects keypoints from the first detected person. If multiple people are in frame, results may be inconsistent.
- **Stable camera:** A tripod eliminates background motion that can confuse velocity-based metrics. Handheld footage with shake will inflate wrist speed readings.

---

## Tuning the Thresholds

All key thresholds are constants at the top of the script and can be adjusted for your specific setup or player level.

```python
# Contact point arm angle thresholds
CONTACT_EARLY_MAX = 80    # raise this if "Early" is triggering too often
CONTACT_LATE_MIN  = 125   # lower this if "Late" is triggering too often

# Kinetic chain sensitivity
HIP_FIRE_DELTA      = 3.0   # lower = more sensitive hip detection
SHOULDER_FIRE_DELTA = 3.0   # lower = more sensitive shoulder detection
WRIST_SPEED_FIRE    = 60.0  # lower if wrist never fires; raise if it fires too early

# Badge display duration
BADGE_FRAMES = 90   # at 30fps this is 3 seconds; set to 60 for 2s or 150 for 5s
```

**If filming from farther away:** Wrist speed values will be smaller (fewer pixels per frame of real movement). Lower `WRIST_SPEED_FIRE` proportionally — try 30–40 for a camera 10+ meters away.

**For junior players:** Consider loosening the contact angle thresholds to `CONTACT_EARLY_MAX = 70` and `CONTACT_LATE_MIN = 135`, since developing players often need a wider ideal window before the pattern is ingrained.

---

## Known Limitations

- **Wrist speed is in pixels/frame, not real-world units.** Comparisons between sessions filmed at different distances or resolutions will not be valid without normalization.
- **Swing phase detection is heuristic.** It is based on horizontal wrist velocity only. Unusual swing styles, defensive slices, or down-the-line forehands may confuse the state machine.
- **Contact point is estimated, not measured.** There is no ball detection. The Acceleration → Follow-Through transition is used as a proxy for contact, which is a close approximation but not exact.
- **Single-player only.** The tool uses keypoints from the first detected person in the frame. For multi-player footage, crop to one player first.
- **Occlusion.** If the hitting arm is occluded by the torso (common when filming from the side on a closed-stance shot), keypoint confidence drops and some metrics may read `--` for those frames.
- **The kinetic chain requires a clean phase transition** from Acceleration to Follow-Through to lock its result. If the phase detector does not detect that transition cleanly, no chain badge will appear for that swing.
