"""Run the trained model live and show raw/stable gesture predictions."""
import os, time
import cv2, joblib, numpy as np
from collections import deque, Counter
from features import extract_features
from vision_utils import create_hand_landmarker, detect_hands
from config import HAND_MODEL_PATH, MODEL_PATH, CONFIDENCE_THRESHOLD, SMOOTHING_WINDOW

if not os.path.isfile(HAND_MODEL_PATH): raise SystemExit(f"Missing {HAND_MODEL_PATH}")
if not os.path.isfile(MODEL_PATH): raise SystemExit("Missing gesture_model.pkl. Run 08_train.py first.")
model = joblib.load(MODEL_PATH)
detector = create_hand_landmarker(HAND_MODEL_PATH)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    detector.close()
    raise SystemExit("Webcam unavailable.")
recent = deque(maxlen=SMOOTHING_WINDOW)
ts = 0
try:
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame,1)
        h,w = frame.shape[:2]
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        ts += 33
        result = detect_hands(detector,rgb,ts)
        raw, conf = "none", 0.0
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            for lm in hand:
                cv2.circle(frame,(int(lm.x*w),int(lm.y*h)),3,(0,255,0),-1)
            features = extract_features(hand)
            if features is not None:
                probs = model.predict_proba(features.reshape(1,-1))[0]
                idx = int(np.argmax(probs))
                raw, conf = str(model.classes_[idx]), float(probs[idx])
                if conf < CONFIDENCE_THRESHOLD: raw = "none"
        recent.append(raw)
        stable = Counter(recent).most_common(1)[0][0]
        cv2.putText(frame,f"RAW: {raw}  confidence: {conf:.2f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,255),2)
        cv2.putText(frame,f"SMOOTHED: {stable}",(10,65),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,0),2)
        cv2.putText(frame,"Q to quit",(10,h-15),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        cv2.imshow("Air Invaders - Live Gesture Prediction",frame)
        if cv2.waitKey(1)&0xFF == ord("q"): break
finally:
    cap.release()
    detector.close()
    cv2.destroyAllWindows()
