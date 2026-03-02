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

## Project structure

```
cv-proj1/
├── main.py                             # main pipeline
├── yolo12n.pt                          # YOLO model for player detection
├── utils/
│   ├── video.py                        # read/write video frames
│   ├── player_utils.py                 # YOLO player detection wrapper
│   ├── ball_utils.py                   # simple ball detection (YOLO)
│   ├── ball_tracker_tracknet.py        # ball tracking with TrackNet
│   ├── tracknet.py                     # TrackNet architecture (VGG-16 encoder-decoder)
│   ├── court_detector_robust.py        # court keypoint detection (deep learning, multi-angle)
│   ├── court_line_detector.py          # legacy court detector (ResNet50)
│   ├── court_line_detector_hough.py    # court detection via Hough transforms
│   ├── homography.py                   # pixel-to-real-world coordinate transform
│   ├── court_zones.py                  # court zone classification (10 zones)
│   └── court_visualizer.py             # top-down court diagrams and trajectory plots
├── output_video/                       # annotated video output
└── runs/detect/                        # YOLO detection artifacts
```

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

The 14 keypoints map to court intersections:
```
0 -------- 4 ----------- 6 -------- 1    far baseline
|          8 ---- 12 --- 9           |    far service line
|          |      |      |           |
|          10 --- 13 --- 11          |    near service line
2 -------- 5 ----------- 7 -------- 3    near baseline
```

The model outputs corners in clockwise order (TL, TR, BR, BL) but our system expects (TL, TR, BL, BR), so corners 2 and 3 get swapped. There's also a fallback that estimates missing near-court doubles corners from the singles sideline geometry.

### Homography
The 14 keypoints are mapped to known real-world tennis court dimensions (10.97m x 23.77m) to compute a homography matrix. This lets us transform any pixel coordinate to real-world meters on the court.

### Shot detection
Shots are detected by finding direction reversals in the ball's y-coordinate trajectory. The algorithm tracks the extreme point (highest or lowest y) and triggers a new shot when the ball reverses direction by more than 2 meters. Consecutive shots going the same direction get merged to reduce noise.

### Zone classification
Each shot landing is classified into one of 10 zones: deuce/ad service boxes, deuce/ad backcourt (near and far side), and left/right alleys. The final zone uses a fusion of ball landing position and receiving player position (60/40 weighting).

## Sample outputs

### Image 1: Annotated video frame
![Annotated video frame](images/image1.jpg)

A single frame from the output video captured during an ATP Tennis TV broadcast of a Djokovic match at the Vienna Open on a blue hard court. The pipeline overlays all detections on the original broadcast feed:
- **Player detection**: YOLO bounding boxes drawn around each detected person with red rectangles and yellow text labels — Player ID: 52 (far-left sideline), Player ID: 40 (far baseline), Player ID: 136 (near-court, mid-rally stance), and additional players/ball kids on the edges.
- **Ball detection**: The tennis ball is identified near center court and labeled "Ball ID: 1" in yellow text.
- **Court keypoints**: All 14 keypoints (numbered 0–13) are plotted as red dots at their detected court-line intersections — points 2, 5 on the near-left baseline, points 4, 6, 8, 12, 9 across the far service line and baseline, points 10, 13, 11 on the near service line, and corners 1, 3 on the right side.
- **Performance overlay**: The right side shows real-time stats including 99% FPS, GPU utilization, GPU temperature, memory clock, and CPU usage.
- The broadcast scoreboard at the bottom reads "Djokovic 1 15".

### Image 2: Shot detection log (console output)
![Shot detection log](images/image2.png)

Terminal console output from the pipeline after processing 300 frames. The log reports "Shots detected: 7" and then lists each shot with full detail:
- **Shot 1** (frames 29–46): Starts at (4.70, 1.15) in far_middle_backcourt, ends at (1.65, 17.92) in near_ad_service_box. Player position at (2.31, 23.36) in near_ad_backcourt. Final zone after 60/40 fusion: **near_ad_service_box**.
- **Shot 2** (frames 49–74): Starts at (2.22, 15.44) in near_ad_service_box, ends at (6.87, 1.44) in far_ad_backcourt. Player at (8.27, 0.40) in far_ad_backcourt. Final: **far_ad_backcourt**.
- **Shot 3** (frames 89–119): Starts at (5.34, 0.39) in far_middle_backcourt, ends at (1.31, 21.41) in left_alley. Final: **left_alley**.
- **Shot 4** (frames 123–147): Starts at (1.51, 16.73) in near_ad_service_box, ends at (4.07, 0.46) in far_deuce_backcourt. Final: **far_deuce_backcourt**.
- **Shot 5** (frames 162–190): Starts at (3.56, 1.04) in far_deuce_backcourt, ends at (2.92, 22.55) in near_ad_backcourt. Final: **near_ad_backcourt**.
- **Shot 6** (frames 194–220): Starts at (3.48, 16.99) in near_ad_service_box, ends at (6.47, 0.46) in far_middle_backcourt. Final: **far_middle_backcourt**.
- **Shot 7** (frames 257–272): Starts at (4.53, 0.99) in far_middle_backcourt, ends at (3.39, 14.78) in near_ad_service_box. Player at (4.31, 23.38) in near_ad_backcourt. Final: **near_ad_service_box**.

GPU monitoring stats are visible on the right side (35°C, 0 RPM fan, 825 mV, 84 MHz memory clock, 4% CPU utilization).

### Image 3: Shot 1 — top-down court trajectory
![Shot 1 trajectory](images/image3.png)

Top-down court diagram for Shot 1, spanning frames 29 to 46. The green rectangle represents the full tennis court with white service lines, baselines, and sidelines drawn to scale.
- **Blue dot** (top-center, frame 29): The shot origin at the far middle backcourt.
- **Orange trajectory line**: The ball's frame-by-frame path, with each point numbered (29, 30, 31, 32, 33, 34, 35, 36, 37, 42, 46). The ball travels from the far baseline downward and slightly to the left, curving through the center of the court.
- **Red dot** (bottom-left, frame 46): The shot landing point near the ad-side baseline (near_ad_service_box).
- **White straight line**: A direct line connecting start to end, showing the overall shot direction — a deep approach shot from the far baseline to the near ad side.

### Image 4: Shot 2 — top-down court trajectory
![Shot 2 trajectory](images/image4.png)

Top-down court diagram for Shot 2, spanning frames 46 to 74. Same green court layout as Shot 1.
- **Blue dot** (bottom-left, frame 46): The shot origin near the ad-side baseline (near_ad_service_box), where Shot 1 landed.
- **Orange trajectory line**: The ball's frame-by-frame path moving diagonally cross-court from bottom-left to top-right. Points are numbered along the path (46, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 71, 72, 74), showing the ball accelerating through the middle of the court.
- **Red dot** (top-right, frame 74): The shot landing point in the far ad backcourt.
- **White straight line**: Direct line from start to end, highlighting the cross-court angle of this return shot — traveling from the near-left baseline diagonally to the far-right backcourt.

## Running it

```bash
cd cv-proj1
python main.py
```

Make sure all three weight files are in the `cv-proj1/` directory. The script reads `test_video.mp4` starting at the 25 second mark, processes up to 300 frames, and outputs:
- `output_video/output.mp4` — annotated video with player boxes, ball tracking, and court keypoints
- `output_video/court_trajectory.png` — top-down court diagram with full ball trajectory
- `output_video/shot_N.png` — individual court diagrams for each detected shot

## Dependencies

- PyTorch
- OpenCV (cv2)
- NumPy
- Matplotlib
- Ultralytics (YOLO)
