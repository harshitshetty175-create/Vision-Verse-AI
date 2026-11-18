"""Shared conversion: 21 hand landmarks -> 42 normalised x/y features."""
import numpy as np

def extract_features(hand_landmarks):
    if hand_landmarks is None or len(hand_landmarks) != 21:
        return None
    points = np.array([[p.x, p.y] for p in hand_landmarks], dtype=np.float32)
    points -= points[0]  # make wrist landmark (0) the origin
    scale = float(np.max(np.abs(points)))
    if scale > 1e-8:
        points /= scale
    return points.flatten()
