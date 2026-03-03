# Tennis Court Vision Analysis

Computer vision pipeline that tracks tennis ball trajectories, detects player positions, and classifies shot landing zones from broadcast video.

## What it does

Takes a tennis match video and:
- Detects players using YOLO
- Tracks the ball using TrackNet (heatmap-based neural network)
- Detects 14 court keypoints using a trained BallTrackerNet variant
- Maps pixel coordinates to real-world court positions via homography
- Identifies shots by detecting direction reversals in the ball trajectory
- Classifies each shot's landing zone (service boxes, backcourt, alleys)
- Outputs an annotated video + court diagrams showing trajectories and shot landings


## Required weights

You need three weight files in `cv-proj1/`:

| File | What it's for |
|------|---------------|
| `yolo12n.pt` | Player detection (YOLO v12 nano) |
| `tracknet_weights.pt` | Ball tracking (TrackNet) |
| `court_detection_weights.pth` | Court keypoint detection (BallTrackerNet, 15-channel output, trained on 8841 images) |

The court detection weights come from the [TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector) project. The file is a full training checkpoint (~6.75GB) — the loader handles extracting just the model weights.

## How it works

### Court keypoint detection
The court detector uses a modified BallTrackerNet (VGG-16 encoder-decoder) with `in_channels=3, out_channels=15`. It outputs a 15-channel heatmap — 14 keypoints + 1 background. Keypoints are extracted by taking the argmax across channels, then using HoughCircles to find the blob center for each keypoint.

The 14 keypoints map to court intersections


The model outputs corners in clockwise order (TL, TR, BR, BL) but our system expects (TL, TR, BL, BR), so corners 2 and 3 get swapped. There's also a fallback that estimates missing near-court doubles corners from the singles sideline geometry.

### Homography
The 14 keypoints are mapped to known real-world tennis court dimensions (10.97m x 23.77m) to compute a homography matrix. This lets us transform any pixel coordinate to real-world meters on the court.

### Shot detection
Shots are detected by finding direction reversals in the ball's y-coordinate trajectory. The algorithm tracks the extreme point (highest or lowest y) and triggers a new shot when the ball reverses direction by more than 2 meters. Consecutive shots going the same direction get merged to reduce noise.

### Zone classification
Each shot landing is classified into one of 10 zones: deuce/ad service boxes, deuce/ad backcourt (near and far side), and left/right alleys. The final zone uses a fusion of ball landing position and receiving player position (60/40 weighting).

## Sample outputs

<img width="1892" height="977" alt="image" src="https://github.com/user-attachments/assets/b5a8afb5-5a36-4af5-868b-33a8d6580702" />


A single frame from the output video captured during an ATP Tennis TV broadcast of a Djokovic match at the Vienna Open on a blue hard court. The pipeline overlays all detections on the original broadcast feed:
- **Player detection**: YOLO bounding boxes drawn around each detected person with red rectangles and yellow text labels — Player ID: 52 (far-left sideline), Player ID: 40 (far baseline), Player ID: 136 (near-court, mid-rally stance), and additional players/ball kids on the edges.
- **Ball detection**: The tennis ball is identified near center court and labeled "Ball ID: 1" in yellow text.
- **Court keypoints**: All 14 keypoints (numbered 0–13) are plotted as red dots at their detected court-line intersections — points 2, 5 on the near-left baseline, points 4, 6, 8, 12, 9 across the far service line and baseline, points 10, 13, 11 on the near service line, and corners 1, 3 on the right side.


<img width="1696" height="753" alt="image" src="https://github.com/user-attachments/assets/bfb1a7b8-c181-4488-9abd-0441bc12a477" />

Showcases rally interpretation shot by shot

<img width="362" height="662" alt="image" src="https://github.com/user-attachments/assets/fd91f5fa-15e4-4f48-ab73-563ce75a7857" />


Top-down court diagram for Shot 1, spanning frames 29 to 46. The green rectangle represents the full tennis court with white service lines, baselines, and sidelines drawn to scale.
- **Blue dot** (top-center, frame 29): The shot origin at the far middle backcourt.
- **Orange trajectory line**: The ball's frame-by-frame path, with each point numbered (29, 30, 31, 32, 33, 34, 35, 36, 37, 42, 46). The ball travels from the far baseline downward and slightly to the left, curving through the center of the court.
- **Red dot** (bottom-left, frame 46): The shot landing point near the ad-side baseline (near_ad_service_box).
- **White straight line**: A direct line connecting start to end, showing the overall shot direction — a deep approach shot from the far baseline to the near ad side.
<img width="376" height="672" alt="image" src="https://github.com/user-attachments/assets/f659b54e-0204-4baa-8f71-d2a4178b37be" />

Top-down court diagram for Shot 2, spanning frames 46 to 74. Same green court layout as Shot 1.
- **Blue dot** (bottom-left, frame 46): The shot origin near the ad-side baseline (near_ad_service_box), where Shot 1 landed.
- **Orange trajectory line**: The ball's frame-by-frame path moving diagonally cross-court from bottom-left to top-right. Points are numbered along the path (46, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 71, 72, 74), showing the ball accelerating through the middle of the court.
- **Red dot** (top-right, frame 74): The shot landing point in the far ad backcourt.
- **White straight line**: Direct line from start to end, highlighting the cross-court angle of this return shot — traveling from the near-left baseline diagonally to the far-right backcourt.

## Running it

```bash
cd cv-proj1
python main.py

