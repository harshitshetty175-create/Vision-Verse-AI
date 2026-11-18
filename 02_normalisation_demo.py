"""Teaching demo: show raw landmark positions and wrist-relative normalised points."""
import os, time
import cv2
import numpy as np
from vision_utils import create_hand_landmarker, detect_hands
from config import HAND_MODEL_PATH

def normalise(points):
    points = points.copy()
    points -= points[0]
    scale = np.max(np.abs(points))
    if scale > 1e-8:
        points /= scale
    return points

if not os.path.isfile(HAND_MODEL_PATH):
    raise SystemExit(f"Missing {HAND_MODEL_PATH}. Put hand_landmarker.task in the models folder.")

detector = create_hand_landmarker(HAND_MODEL_PATH)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    detector.close()
    raise SystemExit("Webcam unavailable.")
ts = 0
try:
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts += 33
        result = detect_hands(detector, rgb, ts)
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            raw = np.array([[p.x, p.y] for p in hand], dtype=np.float32)
            norm = normalise(raw)
            # Left panel: real screen coordinates. Right panel: wrist-relative features.
            left = frame.copy()
            right = np.zeros_like(frame)
            right[:] = (28, 22, 38)
            for i, (x,y) in enumerate(raw):
                px, py = int(x*w), int(y*h)
                cv2.circle(left, (px,py), 4, (0,255,0), -1)
                cv2.putText(left, str(i), (px+4,py-4), cv2.FONT_HERSHEY_SIMPLEX, .35, (255,255,255), 1)
            # scale normalised points into a visible drawing area
            for i, (x,y) in enumerate(norm):
                px = int(w*0.5 + x*w*0.28)
                py = int(h*0.5 + y*h*0.28)
                cv2.circle(right, (px,py), 4, (0,220,255), -1)
            cv2.putText(left, "RAW: screen position changes", (10,25), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 2)
            cv2.putText(right, "NORMALISED: wrist is origin", (10,25), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 2)
            combined = np.hstack([left, right])
            cv2.imshow("Normalisation demo - Q to quit", combined)
        else:
            cv2.putText(frame, "Show one hand", (20,35), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,255), 2)
            cv2.imshow("Normalisation demo - Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    detector.close()
    cv2.destroyAllWindows()
