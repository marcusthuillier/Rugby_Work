#!/usr/bin/env python3
"""
Rugby Pipeline: Video → Frames → Detection → Tracking → Features → Training

Usage:
    python pipeline.py --video videos/my_match.mp4
    python pipeline.py --video videos/my_match.mp4 --frame_rate 2 --field_w 100 --field_h 70

Stages:
    1. Extract frames from video at specified frame_rate
    2. Auto-detect field homography (or load saved calibration)
    3. Run YOLOv8 on every frame for player/ball detection
    4. Auto-assign teams via K-means jersey color clustering
    5. Assign persistent player IDs + compute velocities
    6. Compute features: ball action, possession, scoring, rewards
    7. Extract state tensors and train Actor model
"""

import os
import argparse
import pickle
import glob
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial import distance as scipy_distance
from sklearn.cluster import KMeans
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── Constants ─────────────────────────────────────────────────────────────────
FIELD_W = 100   # Field length in meters (goal line to goal line)
FIELD_H = 70    # Field width in meters
PLAYER_TRACKER_THRESH = 10  # Max distance (meters) to link player across frames
YOLO_MODEL = "yolov8m.pt"
COCO_PERSON_CLASS = 0
COCO_BALL_CLASS = 32


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Frame Extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_frames(video_path, output_dir, frame_rate=2):
    """Extract frames from video at `frame_rate` fps. Saves JPEGs to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(fps / frame_rate))
    frame_idx = 0
    saved = 0

    print(f"[Stage 1] Video fps={fps:.1f}, extracting every {step} frames "
          f"({frame_rate} fps target)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            out_path = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"[Stage 1] Extracted {saved} frames to '{output_dir}'")
    return sorted(glob.glob(os.path.join(output_dir, "*.jpg")))


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Auto Homography
# ══════════════════════════════════════════════════════════════════════════════

def order_points(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL has smallest x+y
    rect[2] = pts[np.argmax(s)]   # BR has largest x+y
    diff = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(diff)]  # TR has smallest y-x
    rect[3] = pts[np.argmax(diff)]  # BL has largest y-x
    return rect


def auto_homography(frame, field_w=FIELD_W, field_h=FIELD_H):
    """
    Automatically compute homography by detecting the green field region.
    Finds the 4 corners of the largest green contour and maps them to
    real-world field coordinates.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Green grass range in OpenCV HSV (H: 0-179, S/V: 0-255)
    lower_green = np.array([30, 30, 30])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean up mask
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError(
            "Could not detect field. Green region not found. "
            "Provide a saved homography matrix at artefact/homography_matrix.pkl"
        )

    field_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(field_contour, True)
    approx = cv2.approxPolyDP(field_contour, epsilon, True).reshape(-1, 2)

    # If approximation gives != 4 points, find 4 extremal corners from convex hull
    if len(approx) != 4:
        hull = cv2.convexHull(field_contour).reshape(-1, 2)
        # Use the 4 corners from bounding approach
        rect = cv2.minAreaRect(hull)
        approx = cv2.boxPoints(rect).astype(np.float32)

    pts_pixel = order_points(approx)
    # Map: TL→(0,0), TR→(field_w,0), BR→(field_w,field_h), BL→(0,field_h)
    pts_world = np.float32([
        [0, 0],
        [field_w, 0],
        [field_w, field_h],
        [0, field_h]
    ])

    H, _ = cv2.findHomography(pts_pixel, pts_world)
    print("[Stage 2] Auto-detected homography from field contour.")
    return H


def get_homography(frame, artefact_path="artefact/homography_matrix.pkl",
                   field_w=FIELD_W, field_h=FIELD_H):
    """Load saved homography or auto-detect and save it."""
    if os.path.exists(artefact_path):
        H = pickle.load(open(artefact_path, "rb"))
        print(f"[Stage 2] Loaded saved homography from '{artefact_path}'")
        return H

    os.makedirs(os.path.dirname(artefact_path), exist_ok=True)
    H = auto_homography(frame, field_w, field_h)
    pickle.dump(H, open(artefact_path, "wb"))
    print(f"[Stage 2] Saved homography to '{artefact_path}'")
    return H


def pixel_to_field(x_px, y_px, H):
    """Transform a single pixel coordinate to real-world field meters."""
    pt = np.array([[[x_px, y_px]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H)
    return float(result[0][0][0]), float(result[0][0][1])


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Player & Ball Detection + Auto Team Assignment
# ══════════════════════════════════════════════════════════════════════════════

def detect_all_frames(frame_paths, H):
    """
    Run YOLOv8 on every frame. Returns raw detections list before team assignment.
    Each entry: {frame_path, frame_num, type, x_px, y_px, x_field, y_field, bbox_crop_bgr}
    """
    yolo = YOLO(YOLO_MODEL)
    raw = []

    print(f"[Stage 3] Running YOLO on {len(frame_paths)} frames...")
    for frame_num, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        results = yolo(frame, verbose=False, conf=0.1)[0]
        boxes = results.boxes

        persons_this_frame = []
        best_ball = None  # Keep only the highest-confidence ball per frame

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())

            if cls == COCO_PERSON_CLASS:
                # Use center-bottom as player position (feet on ground)
                x_px = (xmin + xmax) // 2
                y_px = ymax
                x_field, y_field = pixel_to_field(x_px, y_px, H)

                # Extract jersey color: center crop of bounding box
                h_box = ymax - ymin
                crop = frame[
                    ymin + h_box // 4 : ymin + h_box * 3 // 4,
                    xmin:xmax
                ]
                color = crop.mean(axis=(0, 1)).tolist() if crop.size > 0 else [0, 0, 0]

                persons_this_frame.append({
                    "frame_path": frame_path,
                    "frame_num": frame_num,
                    "type": "person",
                    "x_px": x_px, "y_px": y_px,
                    "x_field": x_field, "y_field": y_field,
                    "color": color
                })

            elif cls == COCO_BALL_CLASS:
                # Keep only the highest-confidence ball per frame
                x_px = (xmin + xmax) // 2
                y_px = (ymin + ymax) // 2
                x_field, y_field = pixel_to_field(x_px, y_px, H)
                if best_ball is None or conf > best_ball["conf"]:
                    best_ball = {
                        "frame_path": frame_path,
                        "frame_num": frame_num,
                        "type": "ball",
                        "conf": conf,
                        "x_px": x_px, "y_px": y_px,
                        "x_field": x_field, "y_field": y_field,
                        "color": [255, 165, 0]
                    }

        raw.extend(persons_this_frame)
        if best_ball is not None:
            raw.append(best_ball)

    print(f"[Stage 3] Detected {len(raw)} objects across all frames.")
    return raw


def assign_teams_kmeans(raw_detections):
    """
    Auto-assign teams using K-means clustering on jersey colors.
    Players are split into 2 clusters → mapped to 'R' and 'L'.
    Team 'R' = cluster with higher mean x_field (right side of pitch).
    Ball entries receive team 'Ball'.
    """
    persons = [d for d in raw_detections if d["type"] == "person"]
    balls = [d for d in raw_detections if d["type"] == "ball"]

    if len(persons) < 2:
        print("[Stage 3] Warning: fewer than 2 players detected, defaulting all to team R")
        for d in persons:
            d["team"] = "R"
        for d in balls:
            d["team"] = "Ball"
            d["object"] = "ball"
        for d in persons:
            d["object"] = "person"
        return raw_detections

    colors = np.array([d["color"] for d in persons], dtype=np.float32)
    n_clusters = min(2, len(persons))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(colors)

    # Determine which cluster is on which side: higher mean x_field → team L (attacks right)
    cluster_x = {}
    for i, d in enumerate(persons):
        c = labels[i]
        cluster_x.setdefault(c, []).append(d["x_field"])
    cluster_mean_x = {c: np.mean(xs) for c, xs in cluster_x.items()}

    # Sort clusters: lower mean x → "R" (attacking from right, currently on left)
    #                higher mean x → "L" (attacking from left, currently on right)
    sorted_clusters = sorted(cluster_mean_x, key=cluster_mean_x.get)
    cluster_to_team = {sorted_clusters[0]: "R", sorted_clusters[1]: "L"} if len(sorted_clusters) == 2 else {sorted_clusters[0]: "R"}

    for i, d in enumerate(persons):
        d["team"] = cluster_to_team.get(labels[i], "R")
        d["object"] = "person"

    for d in balls:
        d["team"] = "Ball"
        d["object"] = "ball"

    team_counts = pd.Series([d["team"] for d in persons]).value_counts().to_dict()
    print(f"[Stage 3] Team assignment: {team_counts}")
    return raw_detections


def build_raw_dataframe(raw_detections):
    """Convert raw detection list to DataFrame matching rugby_detection_raw.csv schema."""
    rows = []
    for d in raw_detections:
        rows.append({
            "x_field": d["x_field"],
            "y_field": d["y_field"],
            "team": d["team"],
            "object": d["object"],
            "color": str(d["color"]),
            "frame": d["frame_path"],
            "frame_num": d["frame_num"],
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — Player ID Tracking + Velocity + Possession
# ══════════════════════════════════════════════════════════════════════════════

def assign_player_ids(df, thresh=PLAYER_TRACKER_THRESH):
    """
    Assign persistent player_id across frames using nearest-neighbor matching.
    Ported from notebook 4 with SettingWithCopyWarning fixed.
    """
    player_tracker = {}  # pid → (x, y, team)
    next_id = 1

    df = df.copy()
    df["player_id"] = pd.array([pd.NA] * len(df), dtype="Int64")

    for frame_num in sorted(df["frame_num"].unique()):
        frame_data = df[df["frame_num"] == frame_num].copy()
        used_ids = set()

        for index, row in frame_data.iterrows():
            if row["object"] != "person":
                continue

            x, y, team = row["x_field"], row["y_field"], row["team"]
            min_dist = float("inf")
            assigned_id = None

            for pid, (px, py, pteam) in player_tracker.items():
                if pteam == team and pid not in used_ids:
                    dist = scipy_distance.euclidean((x, y), (px, py))
                    if dist < min_dist and dist < thresh:
                        min_dist = dist
                        assigned_id = pid

            if assigned_id is None:
                assigned_id = next_id
                next_id += 1

            df.at[index, "player_id"] = assigned_id
            player_tracker[assigned_id] = (x, y, team)
            used_ids.add(assigned_id)

    # Assign a stable player_id=0 to all ball rows so velocity can be computed
    df.loc[df["object"] == "ball", "player_id"] = 0

    df["player_id"] = df["player_id"].astype("Int64")
    print(f"[Stage 4] Assigned {df['player_id'].nunique()} unique player IDs (0 = ball)")
    return df


def compute_velocities_and_possession(df):
    """
    Compute per-player velocity and identify attacker_with_ball per frame.
    SettingWithCopyWarning fixed by using df.loc for all assignments.
    """
    df = df.copy()

    # Velocity via groupby shift
    df["prev_x"] = df.groupby("player_id")["x_field"].shift(1)
    df["prev_y"] = df.groupby("player_id")["y_field"].shift(1)
    df["velocity_x"] = df["x_field"] - df["prev_x"]
    df["velocity_y"] = df["y_field"] - df["prev_y"]

    df["attacker_with_ball"] = 0

    for frame_num in df["frame_num"].unique():
        ball_rows = df[(df["frame_num"] == frame_num) & (df["object"] == "ball")]
        if ball_rows.empty:
            continue

        ball_x = ball_rows.iloc[0]["x_field"]
        ball_y = ball_rows.iloc[0]["y_field"]

        player_rows = df[(df["frame_num"] == frame_num) & (df["object"] == "person")]
        if player_rows.empty:
            continue

        # Compute distances and find closest — fixed: use df.loc, not chained assignment
        distances = np.sqrt(
            (df.loc[player_rows.index, "x_field"] - ball_x) ** 2 +
            (df.loc[player_rows.index, "y_field"] - ball_y) ** 2
        )
        closest_idx = distances.idxmin()
        df.loc[closest_idx, "attacker_with_ball"] = 1

    df["attacker_with_ball"] = df["attacker_with_ball"].fillna(0).astype(int)
    print("[Stage 4] Velocities and ball possession computed.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — Feature Extraction (ported from notebook 6, cells 3–9)
# ══════════════════════════════════════════════════════════════════════════════

def compute_movement_features(df):
    """Compute previous positions, velocity, speed. Ported from notebook 6 cell 3."""
    df = df.copy()
    df["prev_x_field"] = df.groupby("player_id")["x_field"].shift(1)
    df["prev_y_field"] = df.groupby("player_id")["y_field"].shift(1)
    df["v_x"] = df["x_field"] - df["prev_x_field"]
    df["v_y"] = df["y_field"] - df["prev_y_field"]
    df["speed"] = (df["v_x"] ** 2 + df["v_y"] ** 2) ** 0.5
    df.fillna(0, inplace=True)
    return df


def get_direction(v_x, v_y):
    if v_x == 0 and v_y == 0:
        return None
    angle_deg = np.degrees(np.arctan2(v_y, v_x))
    if   -22.5 <= angle_deg < 22.5:   return "Right"
    elif  22.5 <= angle_deg < 67.5:   return "Up-Right"
    elif  67.5 <= angle_deg < 112.5:  return "Up"
    elif 112.5 <= angle_deg < 157.5:  return "Up-Left"
    elif  157.5 <= angle_deg or angle_deg < -157.5: return "Left"
    elif -157.5 <= angle_deg < -112.5: return "Down-Left"
    elif -112.5 <= angle_deg < -67.5:  return "Down"
    else:                              return "Down-Right"


def compute_direction(df):
    df = df.copy()
    df["movement_direction"] = df.apply(
        lambda r: get_direction(r["v_x"], r["v_y"]), axis=1
    )
    return df


def find_player_in_possession(row, frame_df):
    """Return player_id of the player closest to the ball in this frame."""
    if row["team"] != "Ball":
        return None
    players = frame_df[frame_df["team"] != "Ball"].copy()
    if players.empty:
        return None
    dists = np.sqrt(
        (players["x_field"] - row["x_field"]) ** 2 +
        (players["y_field"] - row["y_field"]) ** 2
    )
    return players.loc[dists.idxmin(), "player_id"]


def compute_possession(df):
    df = df.copy()
    df["player_in_possession"] = df.apply(
        lambda r: find_player_in_possession(
            r, df[df["frame_num"] == r["frame_num"]]
        ),
        axis=1
    )
    return df


MAX_BALL_SPEED = 25   # m/frame at 2fps — anything above this is a camera cut / noisy detection
PASS_KICK_THRESHOLD = 15  # m/frame: below = Pass, above = Kick

def classify_ball_action(row):
    if row["team"] != "Ball" or row["speed"] == 0:
        return None
    # Filter out camera cuts and noisy ball detections
    if row["speed"] > MAX_BALL_SPEED:
        return None
    curr = row["player_in_possession"]
    prev = row.get("prev_player_in_possession")
    if pd.isna(curr) or pd.isna(prev):
        return None
    if curr != prev:
        # Possession changed — short distance = Pass, long = Kick
        return "Pass" if row["speed"] < PASS_KICK_THRESHOLD else "Kick"
    # Same player still has the ball — Carry
    return "Carry"


def compute_ball_action(df):
    df = df.copy()
    # Compute previous player_in_possession within ball rows only
    ball_idx = df[df["team"] == "Ball"].sort_values("frame_num").index
    df.loc[ball_idx, "prev_player_in_possession"] = (
        df.loc[ball_idx, "player_in_possession"].shift(1).values
    )
    df["ball_action"] = df.apply(classify_ball_action, axis=1)
    return df


def detect_scoring(row, prev_row, try_left=0, try_right=100,
                   goal_y_min=30, goal_y_max=60):
    if row["team"] != "Ball":
        return None
    if row["player_in_possession"] is not None:
        if row["x_field"] <= try_left or row["x_field"] >= try_right:
            return "Try"
    if row["ball_action"] == "Kick" and prev_row is not None:
        if (goal_y_min <= row["y_field"] <= goal_y_max and (
            (row["x_field"] < prev_row["x_field"] and row["x_field"] <= try_left) or
            (row["x_field"] > prev_row["x_field"] and row["x_field"] >= try_right)
        )):
            return "Drop Goal"
    return None


def detect_turnover(row, prev_row):
    if row["team"] != "Ball":
        return None
    if (prev_row is not None and
            pd.notna(row.get("possession_id")) and
            pd.notna(prev_row.get("possession_id")) and
            row["possession_id"] != prev_row["possession_id"]):
        return "Turnover"
    return None


def compute_events_and_rewards(df):
    """Compute scoring, turnover, and reward. Ported from notebook 6 cells 7–9."""
    df = df.copy()

    # Ensure possession_id exists
    if "possession_id" not in df.columns:
        df["possession_id"] = None

    shifted = df.shift(1)

    df["scoring_event"] = df.apply(
        lambda r: detect_scoring(r, shifted.iloc[r.name]), axis=1
    )
    df["turnover_event"] = df.apply(
        lambda r: detect_turnover(r, shifted.iloc[r.name]), axis=1
    )

    def calc_reward(row):
        reward = 0
        if row["scoring_event"] == "Try":         reward += 10
        elif row["scoring_event"] == "Drop Goal": reward += 5
        if row["ball_action"] == "Pass":          reward += 1
        if pd.notna(row.get("v_x")):
            if row["team"] == "R":
                reward += abs(row["v_x"]) * 0.5 if row["v_x"] < 0 else -abs(row["v_x"]) * 0.25
            elif row["team"] == "L":
                reward += abs(row["v_x"]) * 0.5 if row["v_x"] > 0 else -abs(row["v_x"]) * 0.25
        if row["turnover_event"] == "Turnover":   reward -= 5
        return reward

    df["reward"] = df.apply(calc_reward, axis=1)
    print("[Stage 5] Events and rewards computed.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — State Extraction + Actor Training
# ══════════════════════════════════════════════════════════════════════════════

def extract_state(row, players_df):
    """
    Build a state tensor for a single ball frame.
    Dimensions: 5 (ball) + N_players*6 (per player) + 5 (context)
    """
    state = [
        row["x_field"], row["y_field"], row["v_x"], row["v_y"],
        row["possession_id"] if pd.notna(row.get("possession_id")) else -1
    ]

    for _, p in players_df.iterrows():
        state.extend([
            p["x_field"], p["y_field"],
            p.get("v_x", 0), p.get("v_y", 0),
            1 if p["team"] == "R" else -1,
            1 if p["player_id"] == row.get("player_in_possession") else 0
        ])

    state.append(1 if row.get("turnover_event") == "Turnover" else 0)
    state.append(1 if row.get("ball_action") == "Pass" else 0)
    state.append(1 if row.get("ball_action") == "Carry" else 0)
    state.append(1 if row.get("ball_action") == "Kick" else 0)
    state.append(
        1 if row.get("scoring_event") == "Try"
        else (2 if row.get("scoring_event") == "Drop Goal" else 0)
    )

    return torch.tensor(state, dtype=torch.float32)


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 3),          # Pass / Carry / Kick
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


def build_training_data(df):
    """
    Build (state, label) pairs from real ball frames with a labeled action.
    Returns states tensor, labels tensor, and the state_dim.
    """
    action_map = {"Pass": 0, "Carry": 1, "Kick": 2}
    ball_rows = df[
        (df["team"] == "Ball") &
        (df["ball_action"].isin(action_map.keys()))
    ]

    if ball_rows.empty:
        return None, None, None

    # Determine canonical state_dim from first ball frame
    sample = ball_rows.iloc[0]
    sample_players = df[
        (df["frame_num"] == sample["frame_num"]) & (df["team"] != "Ball")
    ]
    state_dim = extract_state(sample, sample_players).shape[0]
    print(f"[Stage 6] state_dim = {state_dim}  "
          f"(5 ball + {len(sample_players)}×6 player + 5 context)")

    states, labels = [], []
    for _, row in ball_rows.iterrows():
        frame_players = df[
            (df["frame_num"] == row["frame_num"]) & (df["team"] != "Ball")
        ]
        s = extract_state(row, frame_players)
        # Pad or truncate to canonical state_dim
        if s.shape[0] < state_dim:
            s = torch.cat([s, torch.zeros(state_dim - s.shape[0])])
        else:
            s = s[:state_dim]
        states.append(s)
        labels.append(action_map[row["ball_action"]])

    return torch.stack(states), torch.tensor(labels), state_dim


def train_actor(df, num_epochs=200):
    """Train the Actor model on real game state tensors."""
    states_t, labels_t, state_dim = build_training_data(df)

    if states_t is None:
        print("[Stage 6] No labeled ball actions found. "
              "Check that enough frames were processed to detect Pass/Carry/Kick events.")
        return None, None

    model = Actor(state_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    print(f"[Stage 6] Training on {len(states_t)} real game samples for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        probs = model(states_t)
        loss = loss_fn(probs, labels_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | Loss: {loss.item():.4f}")

    print("[Stage 6] Training complete.")
    return model, state_dim


def run_inference(model, state_dim, df):
    """Run inference on the last real game state from the processed dataframe."""
    model.eval()
    labels = ["Pass", "Carry", "Kick"]
    ball_rows = df[df["team"] == "Ball"].sort_values("frame_num")
    if ball_rows.empty:
        print("[Inference] No ball frames found.")
        return
    last_ball = ball_rows.iloc[-1]
    frame_players = df[
        (df["frame_num"] == last_ball["frame_num"]) & (df["team"] != "Ball")
    ]
    state = extract_state(last_ball, frame_players)
    if state.shape[0] < state_dim:
        state = torch.cat([state, torch.zeros(state_dim - state.shape[0])])
    else:
        state = state[:state_dim]
    with torch.no_grad():
        probs = model(state.unsqueeze(0))
        action = torch.argmax(probs, dim=-1).item()
    print(f"\n[Inference] Predicted action: {labels[action]}")
    print(f"[Inference] Probabilities: Pass={probs[0][0]:.3f}  "
          f"Carry={probs[0][1]:.3f}  Kick={probs[0][2]:.3f}")


def predict_from_still(image_path):
    """
    Load the trained model and predict the ball-carrier action from a single image.
    Runs YOLO detection, applies homography, builds the state vector, runs inference.
    """
    base_dir       = os.path.dirname(os.path.abspath(__file__))
    model_path     = os.path.join(base_dir, "artefact", "actor_model.pt")
    homography_path = os.path.join(base_dir, "artefact", "homography_matrix.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError("No trained model found. Run the pipeline on a video first.")
    if not os.path.exists(homography_path):
        raise FileNotFoundError("No homography matrix found. Run the pipeline on a video first.")

    # Load model
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dim = checkpoint["state_dim"]
    model = Actor(state_dim)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Load homography
    with open(homography_path, "rb") as f:
        H = pickle.load(f)

    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Detect with YOLO
    yolo = YOLO("yolov8m.pt")
    results = yolo(frame, verbose=False, conf=0.1)[0]
    boxes = results.boxes

    detections = []
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = yolo.names[cls_id]
        if cls_name not in ("person", "sports ball"):
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        if cls_name == "person":
            px, py = (x1 + x2) / 2, y2
        else:
            px, py = (x1 + x2) / 2, (y1 + y2) / 2
        pt = np.array([[[px, py]]], dtype=np.float32)
        real = cv2.perspectiveTransform(pt, H)[0][0]
        color = frame[int(py), int(px)].tolist() if 0 <= int(py) < frame.shape[0] and 0 <= int(px) < frame.shape[1] else [128, 128, 128]
        detections.append({
            "x_field": float(real[0]), "y_field": float(real[1]),
            "object": "person" if cls_name == "person" else "ball",
            "color": color, "frame_num": 0, "player_id": None,
        })

    if not detections:
        print("[Predict] No players or ball detected in image.")
        return

    df = pd.DataFrame(detections)
    # Assign teams via k-means on jersey color
    persons = df[df["object"] == "person"].copy()
    if len(persons) >= 2:
        colors = np.array(persons["color"].tolist(), dtype=np.float32)
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(colors)
        team_labels = ["R", "L"]
        persons["team"] = [team_labels[l] for l in km.labels_]
        df.loc[persons.index, "team"] = persons["team"]
    df.loc[df["object"] == "ball", "team"] = "Ball"

    # Add zero velocities (single frame — no previous frame to diff)
    df["v_x"] = 0.0
    df["v_y"] = 0.0
    df["speed"] = 0.0

    # Assign player IDs
    pid = 1
    for idx, row in df.iterrows():
        if row["object"] == "person":
            df.at[idx, "player_id"] = pid
            pid += 1
        else:
            df.at[idx, "player_id"] = 0

    # Find player in possession (closest to ball)
    ball_rows = df[df["team"] == "Ball"]
    player_rows = df[df["team"] != "Ball"]
    possession_id = None
    if not ball_rows.empty and not player_rows.empty:
        ball = ball_rows.iloc[0]
        dists = np.sqrt((player_rows["x_field"] - ball["x_field"])**2 +
                        (player_rows["y_field"] - ball["y_field"])**2)
        possession_id = player_rows.loc[dists.idxmin(), "player_id"]

    # Build state from ball row
    if ball_rows.empty:
        print("[Predict] No ball detected — cannot build state.")
        return

    ball_row = ball_rows.iloc[0].copy()
    ball_row["possession_id"] = possession_id
    ball_row["player_in_possession"] = possession_id
    ball_row["turnover_event"] = None
    ball_row["ball_action"] = None
    ball_row["scoring_event"] = None

    state = extract_state(ball_row, player_rows)
    if state.shape[0] < state_dim:
        state = torch.cat([state, torch.zeros(state_dim - state.shape[0])])
    else:
        state = state[:state_dim]

    with torch.no_grad():
        probs = model(state.unsqueeze(0))
        action_idx = torch.argmax(probs, dim=-1).item()

    labels = ["Pass", "Carry", "Kick"]
    print(f"\n[Predict] Image: {os.path.basename(image_path)}")
    print(f"[Predict] Players detected: {len(player_rows)}  |  Ball detected: {not ball_rows.empty}")
    print(f"[Predict] Recommended action: {labels[action_idx]}")
    print(f"[Predict] Probabilities: Pass={probs[0][0]:.3f}  Carry={probs[0][1]:.3f}  Kick={probs[0][2]:.3f}")
    return labels[action_idx], probs[0].tolist()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def process_video_to_df(video_path, frames_dir, artefact_path, frame_rate, field_w, field_h):
    """Run stages 1-5 on a video and return the processed dataframe."""
    frame_paths = extract_frames(video_path, frames_dir, frame_rate)
    if not frame_paths:
        raise RuntimeError("No frames extracted. Check video path.")
    first_frame = cv2.imread(frame_paths[0])
    H = get_homography(first_frame, artefact_path, field_w, field_h)
    raw_detections = detect_all_frames(frame_paths, H)
    raw_detections = assign_teams_kmeans(raw_detections)
    df = build_raw_dataframe(raw_detections)
    df = assign_player_ids(df)
    df = compute_velocities_and_possession(df)
    df = compute_movement_features(df)
    df = compute_direction(df)
    df = compute_possession(df)
    df = compute_ball_action(df)
    df = compute_events_and_rewards(df)
    return df


def run_pipeline(video_path, frame_rate=2, field_w=FIELD_W, field_h=FIELD_H):
    base_dir        = os.path.dirname(os.path.abspath(__file__))
    frames_dir      = os.path.join(base_dir, "frames_raw")
    data_dir        = os.path.join(base_dir, "data", "training")
    model_save_path = os.path.join(base_dir, "artefact", "actor_model.pt")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "artefact"), exist_ok=True)

    # Derive a per-video CSV name from the video filename
    video_stem   = os.path.splitext(os.path.basename(video_path))[0]
    video_stem   = "".join(c if c.isalnum() or c in "-_" else "_" for c in video_stem)
    artefact_path = os.path.join(base_dir, "artefact", f"homography_{video_stem}.pkl")
    processed_csv = os.path.join(data_dir, f"{video_stem}.csv")

    # ── Stages 1-5 ───────────────────────────────────────────────────────────
    df = process_video_to_df(video_path, frames_dir, artefact_path, frame_rate, field_w, field_h)
    df.to_csv(processed_csv, index=False)
    print(f"[Stage 5] Saved {len(df)} rows → '{processed_csv}'")

    # ── Stage 6: Combine all training CSVs and retrain ───────────────────────
    all_csvs = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    combined = pd.concat([pd.read_csv(p) for p in all_csvs], ignore_index=True)
    print(f"[Stage 6] Combined {len(all_csvs)} training file(s) → {len(combined)} total rows")

    model, state_dim = train_actor(combined)
    if model is not None:
        torch.save({"model_state": model.state_dict(), "state_dim": state_dim},
                   model_save_path)
        print(f"[Stage 6] Model saved to '{model_save_path}'")
        run_inference(model, state_dim, combined)

    print("\n✅ Pipeline complete.")
    return combined, model


def evaluate_on_video(video_path, frame_rate=2, field_w=FIELD_W, field_h=FIELD_H):
    """
    Run stages 1-5 on a new video, predict actions with the saved model,
    compare against detected actions, and print a confusion matrix.
    """
    base_dir        = os.path.dirname(os.path.abspath(__file__))
    frames_dir      = os.path.join(base_dir, "frames_raw")
    artefact_path   = os.path.join(base_dir, "artefact", "homography_matrix.pkl")
    model_path      = os.path.join(base_dir, "artefact", "actor_model.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError("No trained model found. Run --video first to train.")

    # Load model
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dim  = checkpoint["state_dim"]
    model      = Actor(state_dim)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"[Eval] Loaded model  state_dim={state_dim}")

    os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "artefact"), exist_ok=True)

    # Stages 1-5 on eval video (homography re-detected fresh)
    eval_artefact = os.path.join(base_dir, "artefact", "homography_eval.pkl")
    frame_paths = extract_frames(video_path, frames_dir, frame_rate)
    if not frame_paths:
        raise RuntimeError("No frames extracted.")

    first_frame = cv2.imread(frame_paths[0])
    H = get_homography(first_frame, eval_artefact, field_w, field_h)

    raw_detections = detect_all_frames(frame_paths, H)
    raw_detections = assign_teams_kmeans(raw_detections)
    df = build_raw_dataframe(raw_detections)
    df = assign_player_ids(df)
    df = compute_velocities_and_possession(df)
    df = compute_movement_features(df)
    df = compute_direction(df)
    df = compute_possession(df)
    df = compute_ball_action(df)
    df = compute_events_and_rewards(df)
    print(f"[Eval] Processed {len(df)} rows from eval video.")

    # Collect labeled ball frames
    action_map = {"Pass": 0, "Carry": 1, "Kick": 2}
    labels_str = ["Pass", "Carry", "Kick"]
    ball_rows = df[
        (df["team"] == "Ball") & (df["ball_action"].isin(action_map.keys()))
    ]

    if ball_rows.empty:
        print("[Eval] No labeled ball actions found in eval video.")
        return

    print(f"[Eval] Found {len(ball_rows)} labeled frames. Running predictions...")

    y_true, y_pred = [], []
    for _, row in ball_rows.iterrows():
        frame_players = df[
            (df["frame_num"] == row["frame_num"]) & (df["team"] != "Ball")
        ]
        state = extract_state(row, frame_players)
        if state.shape[0] < state_dim:
            state = torch.cat([state, torch.zeros(state_dim - state.shape[0])])
        else:
            state = state[:state_dim]
        with torch.no_grad():
            probs = model(state.unsqueeze(0))
            pred  = torch.argmax(probs, dim=-1).item()
        y_true.append(action_map[row["ball_action"]])
        y_pred.append(pred)

    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print("\n" + "═" * 52)
    print("  CONFUSION MATRIX  (rows = actual, cols = predicted)")
    print("═" * 52)
    header = f"{'':>10}  {'Pass':>6}  {'Carry':>6}  {'Kick':>6}"
    print(header)
    print("─" * 52)
    for i, lbl in enumerate(labels_str):
        row_str = f"  {lbl:<8}  " + "  ".join(f"{cm[i][j]:>6}" for j in range(3))
        print(row_str)
    print("═" * 52)

    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, target_names=labels_str, zero_division=0))

    # Overall accuracy
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    print(f"Overall accuracy: {correct}/{len(y_true)} = {correct/len(y_true)*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rugby automated pipeline")
    parser.add_argument("--video",    help="Path to match video — extract, detect, train model")
    parser.add_argument("--evaluate", help="Path to eval video — predict with saved model + confusion matrix")
    parser.add_argument("--predict",  help="Path to a still image — predict action using saved model")
    parser.add_argument("--frame_rate", type=int, default=2,
                        help="Frames per second to extract (default: 2)")
    parser.add_argument("--field_w", type=float, default=FIELD_W,
                        help="Field length in meters (default: 100)")
    parser.add_argument("--field_h", type=float, default=FIELD_H,
                        help="Field width in meters (default: 70)")
    args = parser.parse_args()

    if args.predict:
        predict_from_still(args.predict)
    elif args.evaluate:
        evaluate_on_video(
            video_path=args.evaluate,
            frame_rate=args.frame_rate,
            field_w=args.field_w,
            field_h=args.field_h,
        )
    elif args.video:
        run_pipeline(
            video_path=args.video,
            frame_rate=args.frame_rate,
            field_w=args.field_w,
            field_h=args.field_h,
        )
    else:
        parser.error("Provide --video (train), --evaluate (confusion matrix), or --predict (still image).")
