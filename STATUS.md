# ProLens — Pipeline & System Status

**Generated:** August 16, 2026

---

## Pipeline Overview

```
iPhone Video (.mp4 / HEVC)
        │
        ▼
  ┌─ Phase 0: Court Calibration ──────────────────────┐
  │  YOLO11-pose auto-detect OR manual keypoint drag   │
  │  → homography (pixel ↔ court meters)               │
  └──────────────┬────────────────────────────────────┘
                 │ court_config.json
                 ▼
  ┌─ Phase 1: Player Tracking ────────────────────────┐
  │  RT-DETR-X + BoT-SORT + RTMPose (17 keypoints)    │
  │  + player_filter (near/far per frame)              │
  │  ‖ TrackNetV3 shuttle detection (parallel)         │
  └──────────────┬────────────────────────────────────┘
                 │ tracks_players.json + shuttle_filtered.csv
                 ▼
        ┌────────┼────────────────────┐
        │        │                    │
        ▼        ▼                    ▼
  ┌─ Lane 1 ─┐ ┌─ Lane 3 ──────┐ ┌─ Lane 2 ──────────┐
  │ Phase 2   │ │ MS-TCN        │ │ videoPreprocessor  │
  │ Phase 3   │ │ S1→S2→S3      │ │ Phase 4A/4B/4C     │
  │ (DISABLED)│ │ (ACTIVE)      │ │ (DISABLED)         │
  └─────┬─────┘ └──────┬───────┘ └────────┬───────────┘
        │               │                  │
        └───────┬───────┘──────────────────┘
                ▼
  ┌─ Phase 5: Fusion + Video + Summary ───────────────┐
  │  fuse_results.py → match_analysis_final.json       │
  │  generate_video.py → annotated video               │
  │  match_summary.py → coaching analytics             │
  └───────────────────────────────────────────────────┘
```

---

## Phase-by-Phase Detail

### Phase 0 — Court Calibration

| | |
|---|---|
| **Script** | `backend/services/court_detection.py` (via webapp) or `phase0/court_calibration.py` (standalone) |
| **Model** | `models/court_kp_v6_merged/weights/best.pt` (YOLO11m-pose, 22 keypoints) |
| **Input** | Raw video (.mp4) + `court_calibration.json` (from Court Annotator v5 webapp: `pixel_points` name→[x,y] for up to 32 court keypoints) |
| **Process** | Temporal median frame (removes players) → YOLO keypoint detection → mirror completion → RANSAC homography (min 4 matched keypoints) |

**Outputs:**

| Output File | Contents |
|---|---|
| `court_config.json` | Homography matrices (pixel↔court), court dimensions, pixel keypoints, court corners polygon, service box regions |
| `court_mask.png` | Binary mask of the playable doubles court area |
| `court_overlay.jpg` | Projected court lines overlaid on a video frame (for human review) |

**Court coordinate system:** Origin at near-left doubles corner, x→right (0–6.1m), y→far baseline (0–13.4m), net at y=6.7m.

**Standalone run:**
```bash
conda run -n vjepa2 python phase0/court_calibration.py \
    --input court_calibration.json \
    --video path/to/video.mp4 \
    --frame-idx 100 \
    --output-dir phase0/output
```
Args: `--frame-idx` (default 100, frame for overlay visualization), `--output-dir` (default: same dir as video)

**Via webapp:** Automatic at upload time — user confirms or drags keypoints to correct.

---

### Phase 1 — Player Tracking (+ TrackNetV3 Shuttle Detection)

| | |
|---|---|
| **Scripts** | `phase1/run_tracking.py` → `phase1/player_filter.py` |
| **Models** | RT-DETR-X (`rtdetr-x.pt`), RTMPose-l Body8 (`exp007_rtmpose/models/rtmpose-l_body8_256x192.onnx`) |
| **Input** | Raw video + `court_config.json` |
| **Process** | RT-DETR person detection → court polygon filter → BoT-SORT tracking (shirt-color re-ID + spectator filter) → RTMPose 17-keypoint pose → joint/body angle computation → court-based near/far assignment |

**Run commands:**
```bash
# Full pipeline (detection + tracking + pose + video)
conda run -n vjepa2 python phase1/run_tracking.py \
    --video path/to/video.mp4 \
    --court path/to/court_config.json \
    --output output/ \
    --compact

# JSON only (no output video, faster)
conda run -n vjepa2 python phase1/run_tracking.py \
    --video path/to/video.mp4 \
    --court path/to/court_config.json \
    --tracks-only --compact

# Player filter (assign near/far roles)
conda run -n vjepa2 python phase1/player_filter.py \
    --config player_config.json \
    --tracks output/{stem}_tracks.json \
    --output output/{stem}_tracks_players.json
```

**TrackNetV3 shuttle detection (runs in parallel with Phase 1):**
```bash
conda run -n vjepa2 python phase2/run_shuttle.py \
    --video path/to/video.mp4 \
    --court path/to/court_config.json \
    --output output/{stem}_shuttle_filtered.csv
```

**Outputs:**

| Output File | Contents |
|---|---|
| `{stem}_tracks.json` | Per-frame: all detected persons with bbox, foot_point, track_id, confidence |
| `{stem}_tracks_players.json` | Per-frame: filtered 1 near + 1 far player with 17 COCO keypoints, 8 joint angles, body angles (torso_lean, trunk_flexion, etc.) |
| `{stem}_tracked.mp4` | Debug video with tracking boxes, IDs, pose skeletons, court overlay |
| `{stem}_tracked_web.mp4` | Web-optimized 960px version |
| `{stem}_shuttle_filtered.csv` | CSV: `Frame,Visibility,X,Y` (Visibility=1 means shuttle detected) |

**Diagnostic tools:**
```bash
# Validation (N sample frames, 3-panel images)
conda run -n vjepa2 python phase1/validate.py \
    --video path/to/video.mp4 --court path/to/court_config.json --n 5

# Dashboard video (multi-panel viz from filtered tracks)
conda run -n vjepa2 python phase1/run_dashboard.py \
    --video path/to/video.mp4 \
    --tracks output/{stem}_tracks_players.json \
    --court path/to/court_config.json

# Add shuttle trail overlay to tracked video
conda run -n vjepa2 python phase1/add_shuttle_trail.py \
    --video output/{stem}_tracked.mp4 \
    --shuttle path/to/{stem}_shuttle_filtered.csv
```

---

### Phase 2 — Rally Segmentation (Lane 1 — DISABLED)

| | |
|---|---|
| **Script** | `phase2/run_rally_segmentation.py` |
| **Input** | `{stem}_tracks_players.json` + `court_config.json` + video (optional, enables frame differencing + overlays) |
| **Process** | Dual-channel motion energy (break score + rally energy from 12+ sub-signals: pose, court-space displacement, wrist velocity, frame differencing) → break-first rally segmentation with dual thresholds → server detection |

**Run commands:**
```bash
# Main rally segmentation
conda run -n vjepa2 python phase2/run_rally_segmentation.py \
    --tracks phase1/output/{stem}_tracks_players.json \
    --court phase0/court_config.json \
    --video path/to/video.mp4

# Evaluation against ground truth
conda run -n vjepa2 python phase2/eval_rule_based.py --video IMG_3525
```

**Outputs:**

| Output File | Contents |
|---|---|
| `{stem}_energy.json` | Per-frame motion energy time series (break_score, rally_energy, state_score, all sub-signals) |
| `{stem}_rallies.json` | Rally segments: start/end frame+sec, server (near/far), confidence, energy stats |
| `{stem}_energy_plot.png` | 5-panel timeline (dual channels, displacement, distance, speed, posture) |
| `{stem}_rallies_overlay.mp4` | Video with rally state banner, energy bar, rally counter, timeline |
| `{stem}_shuttle_trail.mp4` | Shuttle trail video overlay (cyan=in flight, orange=on ground, red=at net) |

**Key modules:** `motion_energy.py` (dual-channel signals), `rally_detector.py` (break-first segmentation), `rally_plot.py` (5-panel matplotlib), `rally_video_overlay.py` (overlay), `shuttle_detector.py` (TrackNetV3 wrapper), `court_skeleton_video.py` (debug wireframe)

**Status:** Disabled — MS-TCN S1/S2 (Lane 3) now handles rally segmentation. Kept for evaluation/debugging.

---

### Phase 3 — Shot Detection & Classification (Lane 1 — DISABLED)

| | |
|---|---|
| **Script** | `phase3/run_shot_classification.py` (orchestrator) |
| **Input** | `{stem}_rallies.json` (Phase 2) + `{stem}_tracks_players.json` (Phase 1) + `{stem}_shuttle_filtered.csv` (Phase 2) + `court_config.json` + video |
| **Process** | 3-signal fusion shot detection (shuttle reversal + wrist spike + shuttle-player proximity) → 33-frame clip extraction → decision-tree rule classification (14 shot types) |

**Components:**

| Module | Purpose |
|---|---|
| `shot_detector.py` | 3-signal fusion: shuttle direction reversal, wrist angular velocity spike, shuttle-player proximity. Serves: wrist spike alone in first 20 frames. Rally shots: 2+ signals within ±3 frames. NMS with MIN_SHOT_GAP=20 frames |
| `clip_extractor.py` | Extracts 33-frame video clips (±16 around contact) + JSON sidecar with pose timeline |
| `extract_shot_clips.py` | Alternative extractor for MS-TCN output: ffmpeg H.264, 960px, CRF 28, faststart |
| `rule_classifier.py` | Decision-tree: arm height, shuttle trajectory/speed, body angle → 14 shot types |

**Run command:**
```bash
conda run -n vjepa2 python phase3/run_shot_classification.py \
    --video path/to/video.mp4 \
    --rallies phase2/output/{stem}_rallies.json \
    --tracks phase1/output/{stem}_tracks_players.json \
    --shuttle phase2/output/{stem}_shuttle_filtered.csv \
    --court phase0/court_config.json
```

**Outputs:**

| Output File | Contents |
|---|---|
| `{stem}_shots.json` | Raw shot detections (contact frame, player, signal scores) |
| `{stem}_shots_rule.json` | Rule-classified shots (shot type, arm height, trajectory) |
| `{stem}_match_analysis.json` | Final merged analysis |
| `output/clips/` | Per-shot .mp4 + .json sidecars |

**Status:** Disabled — MS-TCN cascade (Lane 3) handles detection and classification. `extract_shot_clips.py` still used for clip generation from MS-TCN output.

---

### Lane 3: MS-TCN Cascade (ACTIVE — Rally + Shot + Classification)

| | |
|---|---|
| **Script** | `experiments/exp007_rtmpose/mstcn_inference.py` |
| **Models** | `experiments/exp007_rtmpose/models/ms_tcn_final_stage1.pt` (S1), `ms_tcn_final_stage3.pt` (S3) |
| **Input** | `{stem}_tracks_players.json` + `court_config.json` + optional `{stem}_shuttle_filtered.csv` |
| **Process** | Compute 284-ch feature vector per frame → S1 break/active → S2 rally/shot boundaries → S3 12-class shot type |
| **Output** | `{stem}_mstcn_shots.json` — rallies[], shots[] with shot_type and confidence |

**Run command (full cascade):**
```bash
conda run -n vjepa2 python experiments/exp007_rtmpose/mstcn_inference.py \
    --tracks path/to/{stem}_tracks_players.json \
    --court  path/to/court_config.json \
    --shuttle path/to/{stem}_shuttle_filtered.csv \
    --output path/to/{stem}_mstcn_shots.json
```

**Run command (stage-3-only, reclassify existing shots):**
```bash
conda run -n vjepa2 python experiments/exp007_rtmpose/mstcn_inference.py \
    --tracks  path/to/{stem}_tracks_players.json \
    --court   path/to/court_config.json \
    --shots   path/to/{stem}_shots.json \
    --stage3-only \
    --output  path/to/{stem}_classified_shots.json
```

**Feature channels (284 total):**

| Group | Channels | Content |
|-------|----------|---------|
| Near player | 113 | positions, velocities, angles |
| Far player | 113 | positions, velocities, angles |
| Inter-player | 4 | relative distance, court positions |
| Biomechanical | 24 | body lean, knee flexion, swing phase |
| Composite | 13 | break/rally + receiver-reaction |
| Sequence context | 12 | 11 prev_shot one-hot + 1 shot_number_norm |
| Shuttle | 5 | detected, x, y, vx, vy from TrackNetV3 |

**Shot taxonomy (12 types):**
```
SERVES:   low_serve  high_serve  flick_serve  drive_serve
OVERHEAD: clear  smash  drop
NET:      net_shot  net_kill
OTHERS:   lift  drive  block
```

**Performance (exp007 final):**

| Stage | Task | F1 |
|-------|------|----|
| S1 | break/active | 0.786 |
| S2 | rally/shot boundaries | 0.907 |
| S3 | shot type (12-class) | 0.551 |

Viterbi enforces player alternation (0 violations).

**Debug artifacts:**
```bash
# Per-shot video clips
conda run -n vjepa2 python phase3/extract_shot_clips.py \
    --shots  path/to/{stem}_mstcn_shots.json \
    --video  path/to/video.mp4 \
    --output path/to/clips/

# Timeline chart
conda run -n vjepa2 python experiments/exp007_rtmpose/generate_timeline.py \
    --shots  path/to/{stem}_mstcn_shots.json \
    --output path/to/mstcn_timeline.png
```

---

### videoPreprocessor — Player Crop Extraction (Lane 2 — DISABLED)

| | |
|---|---|
| **Script** | `videoPreprocessor/` |
| **Input** | Raw video + `court_config.json` + `{stem}_tracks_players.json` |
| **Process** | Crops near/far player views (384×384) from raw video using court homography + player tracking data |
| **Output** | `raw.mp4`, `near.mp4`, `far.mp4`, `meta.json` per video (in `training_bank/`) |

**Additional tools:**
- `VideoAnnotator.html` — Offline annotation tool for ground truth labeling
- `clip_extractor_gt.py` — Extracts fixed-length clips from ground truth annotations
- `train_shot_probe.py` — V-JEPA 2 shot classification training
- `inference.py` — Phase 4C inference

**Status:** Disabled. Needed for V-JEPA 2 training/inference (Lane 2). Requires 5+ annotated videos.

---

### Phase 4A — V-JEPA 2 Rally Detection (Lane 2 — DISABLED)

| | |
|---|---|
| **Scripts** | `phase4/phase4a_rally_detection/` — `prepare_rally_data.py` → `train_rally_probe.py` → `inference_rally.py` |
| **Model** | Frozen V-JEPA 2 ViT-L (`checkpoints/vitl.pt`) + trainable AttentiveProbe |
| **Input** | 3 video files (raw + near-crop + far-crop) + `ground_truth.json` |
| **Process** | Multi-view fusion: sliding-window binary classification (rally vs break) with frozen ViT-L encoder |
| **Output** | `rallies_ai.json` — rally/break segments with frame ranges, timestamps, confidence |

**Run commands:**
```bash
# Prepare training data
conda run -n vjepa2 python phase4/phase4a_rally_detection/prepare_rally_data.py \
    --ground-truth videoPreprocessor/ground_truth.json \
    --raw-video video.mp4 \
    --near-video videoPreprocessor/output/{stem}_near.mp4 \
    --far-video videoPreprocessor/output/{stem}_far.mp4 \
    --output-dir phase4/phase4a_rally_detection/datasets/{stem}/

# Train
conda run -n vjepa2 python phase4/phase4a_rally_detection/train_rally_probe.py \
    --data phase4/phase4a_rally_detection/datasets/{stem}/rally_windows.json \
    --output-dir phase4/phase4a_rally_detection/runs/rally_v1/ --epochs 30

# Inference
conda run -n vjepa2 python phase4/phase4a_rally_detection/inference_rally.py \
    --raw-video video.mp4 \
    --near-video videoPreprocessor/output/{stem}_near.mp4 \
    --far-video videoPreprocessor/output/{stem}_far.mp4 \
    --model phase4/phase4a_rally_detection/runs/rally_v1/best.pt \
    --output rallies_ai.json

# Visualize
conda run -n vjepa2 python phase4/phase4a_rally_detection/visualize_rally.py \
    --video video.mp4 --rallies rallies_ai.json --output rally_overlay.mp4
```

---

### Phase 4B — V-JEPA 2 Shot Moment Detection (Lane 2 — DISABLED)

| | |
|---|---|
| **Scripts** | `phase4/phase4b_shot_detection/` — `prepare_shot_data.py` → `train_shot_detector.py` → `inference_shots.py` |
| **Model** | Frozen V-JEPA 2 ViT-L + trainable AttentiveProbe (3-class: no_shot / near_shot / far_shot) |
| **Input** | Rally boundaries (from Phase 4A or ground truth) + 3 video files |
| **Process** | Within rally segments, sliding-window 3-class classification + peak detection for shot moments |
| **Output** | `shots_ai.json` — detected shots with contact_frame, player, confidence, clip ranges |

**Run commands:**
```bash
# Prepare training data (3:1 negative ratio balancing)
conda run -n vjepa2 python phase4/phase4b_shot_detection/prepare_shot_data.py \
    --ground-truth videoPreprocessor/ground_truth.json \
    --raw-video video.mp4 \
    --near-video videoPreprocessor/output/{stem}_near.mp4 \
    --far-video videoPreprocessor/output/{stem}_far.mp4 \
    --output-dir phase4/phase4b_shot_detection/datasets/{stem}/

# Train
conda run -n vjepa2 python phase4/phase4b_shot_detection/train_shot_detector.py \
    --data phase4/phase4b_shot_detection/datasets/{stem}/shot_windows.json \
    --output-dir phase4/phase4b_shot_detection/runs/shot_det_v1/ --epochs 30

# Inference
conda run -n vjepa2 python phase4/phase4b_shot_detection/inference_shots.py \
    --rallies phase4/phase4a_rally_detection/rallies_ai.json \
    --raw-video video.mp4 \
    --near-video videoPreprocessor/output/{stem}_near.mp4 \
    --far-video videoPreprocessor/output/{stem}_far.mp4 \
    --model phase4/phase4b_shot_detection/runs/shot_det_v1/best.pt \
    --output shots_ai.json
```

---

### Phase 5 — Fusion + Annotated Video + Coaching Summary

| | |
|---|---|
| **Scripts** | `phase5/fuse_results.py` → `phase5/generate_video.py` → `phase5/match_summary.py` |
| **Input** | Any combination of lane outputs (currently lane 3 only: `{stem}_mstcn_shots.json`) |
| **Process** | Rally fusion (Union-Find clustering, IoU ≥ 0.3) → shot fusion (±15 frame tolerance) → BWF rally-point scoring → annotated video → coaching analytics |

**Run commands:**
```bash
# Fusion
conda run -n vjepa2 python phase5/fuse_results.py \
    --mstcn-shots path/to/{stem}_mstcn_shots.json \
    --output path/to/match_analysis_final.json \
    --fps 30

# Annotated video
conda run -n vjepa2 python phase5/generate_video.py \
    --input  path/to/match_analysis_final.json \
    --video  path/to/video.mp4 \
    --output path/to/{stem}_annotated.mp4

# Coaching summary (console + optional JSON)
conda run -n vjepa2 python phase5/match_summary.py \
    --input path/to/match_analysis_final.json \
    --json  path/to/summary.json
```

**Fusion supports all lanes when enabled:**
```bash
conda run -n vjepa2 python phase5/fuse_results.py \
    --rule-rallies  path/to/{stem}_rallies.json \
    --ai-rallies    path/to/rallies_ai.json \
    --mstcn-shots   path/to/{stem}_mstcn_shots.json \
    --rule-shots    path/to/{stem}_shots_rule.json \
    --ai-shots      path/to/shots_ai.json \
    --ground-truth  path/to/ground_truth.json \
    --output path/to/match_analysis_final.json
```

**Outputs:**

| Output File | Contents |
|---|---|
| `match_analysis_final.json` | Fused rallies, fused shots (with classifications), break segments, BWF rally-point scoring, confidence breakdown, summary stats |
| `{stem}_annotated.mp4` | Video with rally/break status bar, shot flash indicators, shot type labels, player shot counts, shot sequence, timeline bar |
| `{stem}_annotated_h264.mp4` | Browser-compatible H.264 re-encode |
| `summary.json` | Per-player profiles (shot types, FH/BH, attack/defense ratio, serve patterns, direction, court side), rally patterns, coaching insights |

**Orchestrator (chains all phases):**
```bash
conda run -n vjepa2 python phase5/run_full_pipeline.py \
    --video path/to/video.mp4 \
    --court path/to/court_config.json \
    --output-dir output/
```
Supports `--skip-phase1`, `--skip-phase2`, `--skip-phase3`, `--skip-preprocess`, `--skip-phase4a`, `--skip-phase4b`, `--skip-phase4c` for partial reruns. Requires Phase 0 (court config) done beforehand.

---

## Webapp

### Architecture

- **Backend:** FastAPI (`backend/main.py`), port 8000
- **Frontend:** Single-page React app served as static HTML (`prolens-dev-v4.html` for dev)
- **State:** In-memory `JobManager` with JSON persistence to `job_states/`

### How to Run

```bash
# Start backend (from project root)
cd /home/ubuntu/vjepa2/prolens/backend
conda run -n vjepa2 uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend: open prolens-dev-v4.html in browser
# Set API IP in localStorage: prolens_ip = "<server-ip>"
# Or use production frontend from frontend/dist/ (auto-served by FastAPI)
```

### API Endpoints

| Method | Endpoint | Description | State Transition |
|--------|----------|-------------|------------------|
| `GET` | `/api/health` | Health check + GPU info | — |
| `POST` | `/api/upload/video` | Upload video → transcode → metadata | → `UPLOADED` |
| `GET` | `/api/upload/{job_id}/thumbnail` | First-frame JPEG | — |
| `GET` | `/api/upload/{job_id}/frame/{n}` | Extract frame N as JPEG | — |
| `POST` | `/api/court/{job_id}/detect` | YOLO auto court detection | → `COURT_DETECTED` |
| `POST` | `/api/court/{job_id}/confirm` | Confirm auto-detected court | → `COURT_CONFIRMED` |
| `POST` | `/api/court/{job_id}/correct` | User-corrected keypoints | → `COURT_CONFIRMED` |
| `GET` | `/api/court/{job_id}/median_frame` | Temporal-median frame JPEG | — |
| `GET` | `/api/court/{job_id}/players` | RT-DETR player detection | — |
| `POST` | `/api/players/{job_id}/name` | Assign player names + near/far | → `PLAYERS_NAMED` |
| `POST` | `/api/pipeline/{job_id}/start` | Start full pipeline | → `PROCESSING` |
| `GET` | `/api/pipeline/{job_id}/status` | Poll job status | — |
| `GET` | `/api/pipeline/{job_id}/stream` | SSE real-time progress | — |
| `GET` | `/api/results/{job_id}` | Final match analysis JSON | — |
| `GET` | `/api/results/{job_id}/video` | Annotated video MP4 | — |
| `GET` | `/api/results/{job_id}/tracked-video` | Phase 1 tracking debug video | — |
| `GET` | `/api/results/{job_id}/shots/{idx}/clip` | Per-shot video clip | — |
| `GET` | `/api/results/{job_id}/mstcn-timeline` | MS-TCN timeline chart PNG | — |
| `GET` | `/api/results/{job_id}/heatmap/{player}` | Player court heatmap PNG | — |
| `GET` | `/api/results/{job_id}/rallies/{id}/clip` | Rally video clip | — |

### User Flow (webapp)

```
1. Upload video          POST /api/upload/video
2. Auto-detect court     POST /api/court/{job_id}/detect
3. Confirm/correct court POST /api/court/{job_id}/confirm or /correct
4. Detect players        GET  /api/court/{job_id}/players
5. Name players          POST /api/players/{job_id}/name
6. Start pipeline        POST /api/pipeline/{job_id}/start
7. Watch progress        GET  /api/pipeline/{job_id}/stream (SSE)
8. View results          GET  /api/results/{job_id}
```

### Job State Machine

```
UPLOADED → COURT_DETECTED → COURT_CONFIRMED → PLAYERS_NAMED → PROCESSING → COMPLETED
                                                                          → FAILED
```

---

## Disk Layout Per Job

```
uploads/{job_id}/
├── source.mp4                    ← transcoded H.264 input
├── thumbnail.jpg                 ← first frame
├── player_thumbs/                ← player crop images
└── corrections/                  ← user court corrections

results/{job_id}/
├── court_config.json             ← homography + court dimensions
├── player_config.json            ← near/far player track assignments
├── phase1/
│   ├── {stem}_tracks.json            ← all detected persons
│   ├── {stem}_tracks_players.json    ← filtered near + far player
│   ├── {stem}_tracked.mp4            ← tracking overlay (may have shuttle trail)
│   └── {stem}_tracked_web.mp4        ← 960px web-optimized version
├── phase2/
│   └── {stem}_shuttle_filtered.csv   ← TrackNetV3 shuttle positions
├── phase3/
│   └── clips/                        ← per-shot video clips (mstcn_shot_NNN.mp4)
├── mstcn_track/
│   ├── {stem}_mstcn_shots.json       ← rallies + shots + classification
│   └── mstcn_timeline.png            ← prediction timeline chart
├── match_analysis_final.json         ← fused final output (served to frontend)
├── {stem}_annotated.mp4              ← annotated video (mp4v codec)
└── {stem}_annotated_h264.mp4         ← browser-compatible re-encode
```

---

## Pipeline Status

### Active Components

| Component | Script | Model | Status |
|-----------|--------|-------|--------|
| Court detection | `backend/services/court_detection.py` | `court_kp_v6_merged` | Active |
| Player tracking | `phase1/run_tracking.py` | RT-DETR-X + RTMPose | Active |
| Player filter | `phase1/player_filter.py` | — | Active |
| Shuttle detection | `phase2/run_shuttle.py` | TrackNetV3 + InpaintNet | Active (parallel) |
| Shuttle trail overlay | `phase1/add_shuttle_trail.py` | — | Active (background) |
| MS-TCN cascade | `exp007_rtmpose/mstcn_inference.py` | S1 + S3 models | Active (sole source) |
| Shot clips | `phase3/extract_shot_clips.py` | — | Active (non-fatal) |
| Fusion | `phase5/fuse_results.py` | — | Active (lane3-only) |
| Annotated video | `phase5/generate_video.py` | — | Active |
| Match summary | `phase5/match_summary.py` | — | Active |

### Disabled Lanes

| Lane | Components | Why Disabled | Re-enable When |
|------|-----------|--------------|----------------|
| Lane 1 (rule-based) | Phase 2 rally seg → Phase 3 shot detection/classification | MS-TCN cascade handles all three stages | Useful for evaluation/debugging |
| Lane 2 (V-JEPA 2) | videoPreprocessor → Phase 4A rally → Phase 4B shot → Phase 4C classification | Needs 5+ annotated videos (currently have ~2) | Annotate more videos (biggest lever: 433s video d5a59a67) |

### Known Issues

1. `phase5/generate_video.py` lines 329–418: double encode (mp4v + ffmpeg) — should be single ffmpeg pipe
2. `build_training_bank.py` not yet written (needed for exp002 retrain)
3. MS-TCN S3 double-triggering from label flickering still present
4. `net_kill` F1=0.000 (only 45 test frames — needs more training data)
5. `phase1/validate.py` still references old MediaPipe PoseEstimator API (errors on pose; works with `--no-pose`)

---

## Key Model Locations

```
models/court_kp_v6_merged/weights/best.pt                    ← YOLO court keypoints (22 kp)
phase1/rtdetr-x.pt                                           ← RT-DETR person detection
experiments/exp007_rtmpose/models/rtmpose-l_body8_256x192.onnx ← RTMPose pose estimation
phase2/TrackNetV3/ckpts/TrackNet_best.pt                     ← shuttle detection
phase2/TrackNetV3/ckpts/InpaintNet_best.pt                   ← shuttle inpainting
experiments/exp007_rtmpose/models/ms_tcn_final_stage1.pt     ← MS-TCN S1 (break/active)
experiments/exp007_rtmpose/models/ms_tcn_final_stage3.pt     ← MS-TCN S3 (shot type)
experiments/exp007_rtmpose/models/training_stats.npz         ← feature normalization stats
checkpoints/vitl.pt                                          ← V-JEPA 2 ViT-L encoder (frozen)
```

---

## Quick Start (Full Pipeline via CLI)

```bash
# 1. Court calibration (or use webapp)
conda run -n vjepa2 python phase0/court_calibration.py \
    --input phase0/court_calibration.json --video video.mp4

# 2. Player tracking
conda run -n vjepa2 python phase1/run_tracking.py \
    --video video.mp4 --court phase0/output/court_config.json --output output/ --compact

# 3. Player filter
conda run -n vjepa2 python phase1/player_filter.py \
    --config phase1/player_config.json \
    --tracks output/{stem}_tracks.json \
    --output output/{stem}_tracks_players.json

# 4. Shuttle detection (can run in parallel with steps 2-3)
conda run -n vjepa2 python phase2/run_shuttle.py \
    --video video.mp4 --court phase0/output/court_config.json \
    --output output/{stem}_shuttle_filtered.csv

# 5. MS-TCN cascade
conda run -n vjepa2 python experiments/exp007_rtmpose/mstcn_inference.py \
    --tracks output/{stem}_tracks_players.json \
    --court  phase0/output/court_config.json \
    --shuttle output/{stem}_shuttle_filtered.csv \
    --output output/{stem}_mstcn_shots.json

# 6. Fusion
conda run -n vjepa2 python phase5/fuse_results.py \
    --mstcn-shots output/{stem}_mstcn_shots.json \
    --output output/match_analysis_final.json

# 7. Annotated video
conda run -n vjepa2 python phase5/generate_video.py \
    --input output/match_analysis_final.json \
    --video video.mp4 \
    --output output/{stem}_annotated.mp4

# 8. Coaching summary
conda run -n vjepa2 python phase5/match_summary.py \
    --input output/match_analysis_final.json
```

**Or use the orchestrator (skips disabled lanes):**
```bash
conda run -n vjepa2 python phase5/run_full_pipeline.py \
    --video video.mp4 \
    --court phase0/output/court_config.json \
    --output-dir output/
```
