
"""
VisionVerse - Clean Hand Gesture Recognition

Gestures:
FIST, OPEN PALM, PEACE, POINT, PINCH

Features:
- Two-hand detection
- 21 hand landmarks
- Live gesture recognition
- Individual gesture counters
- Clean camera interface
- No black dashboard backgrounds

Run:
py -3.13 02_hand_tracking.py
"""

from pathlib import Path
import math
import time
from collections import Counter

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BASE_DIR = Path(__file__).resolve().parent

MODEL_CANDIDATES = [
    BASE_DIR / "hand_landmarker.task",
    BASE_DIR / "models" / "hand_landmarker.task",
]

MODEL_PATH = next(
    (p for p in MODEL_CANDIDATES if p.is_file()), None
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

GESTURES = ["FIST", "OPEN PALM", "PEACE", "POINT", "PINCH"]


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def finger_extended(landmarks, mcp, pip, tip):
    wrist = landmarks[0]

    tip_distance = distance(wrist, landmarks[tip])
    pip_distance = distance(wrist, landmarks[pip])

    return tip_distance > pip_distance * 1.12


def thumb_extended(landmarks):
    wrist = landmarks[0]

    tip_distance = distance(wrist, landmarks[4])
    ip_distance = distance(wrist, landmarks[3])

    return tip_distance > ip_distance * 1.15


def classify_gesture(landmarks):
    thumb = thumb_extended(landmarks)

    index = finger_extended(landmarks, 5, 6, 8)
    middle = finger_extended(landmarks, 9, 10, 12)
    ring = finger_extended(landmarks, 13, 14, 16)
    pinky = finger_extended(landmarks, 17, 18, 20)

    pinch_distance = distance(landmarks[4], landmarks[8])

    hand_scale = (
        distance(landmarks[0], landmarks[9]) +
        distance(landmarks[5], landmarks[17])
    ) / 2

    hand_scale = max(hand_scale, 0.001)
    pinch_ratio = pinch_distance / hand_scale

    if pinch_ratio < 0.32:
        return "PINCH"

    if not any([thumb, index, middle, ring, pinky]):
        return "FIST"

    if all([thumb, index, middle, ring, pinky]):
        return "OPEN PALM"

    if index and middle and not ring and not pinky:
        return "PEACE"

    if index and not middle and not ring and not pinky:
        return "POINT"

    return "OTHER"


def landmark_xy(landmark, width, height):
    x = max(0, min(width - 1, int(landmark.x * width)))
    y = max(0, min(height - 1, int(landmark.y * height)))
    return x, y


def draw_clean_text(frame, text, position, color=(255, 255, 255),
                    scale=0.65):
    """
    Draw readable text directly over the camera.
    No filled background or black panel.
    """
    cv2.putText(
        frame, text, position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, (25, 25, 25), 3, cv2.LINE_AA
    )

    cv2.putText(
        frame, text, position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, 2, cv2.LINE_AA
    )


def open_camera():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)

    if camera.isOpened():
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return camera


def main():

    if MODEL_PATH is None:
        raise SystemExit(
            "hand_landmarker.task not found. "
            "Place the model inside the project folder."
        )

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.60,
        min_hand_presence_confidence=0.60,
        min_tracking_confidence=0.60,
    )

    detector = vision.HandLandmarker.create_from_options(options)
    camera = open_camera()

    if not camera.isOpened():
        detector.close()
        raise SystemExit("Cannot open webcam.")

    counts = Counter()
    stable_states = {}

    STABLE_FRAMES = 5

    start_time = time.perf_counter()
    last_timestamp = -1

    window_name = "VisionVerse - Gesture Recognition"

    print("VisionVerse Gesture Recognition started.")
    print("Press Q or ESC to exit.")

    try:
        while True:

            ok, frame = camera.read()

            if not ok:
                print("Camera frame unavailable.")
                break

            frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = max(
                int((time.perf_counter() - start_time) * 1000),
                last_timestamp + 1
            )

            last_timestamp = timestamp_ms

            result = detector.detect_for_video(
                mp_image, timestamp_ms
            )

            detected_count = len(result.hand_landmarks)

            # Clean hand count - no background rectangle
            draw_clean_text(
                frame,
                f"HANDS DETECTED: {detected_count}",
                (20, 45),
                (80, 240, 130),
                0.85
            )

            active_keys = set()

            for hand_index, landmarks in enumerate(
                result.hand_landmarks
            ):

                points = [
                    landmark_xy(lm, width, height)
                    for lm in landmarks
                ]

                # Draw hand skeleton
                for start, end in HAND_CONNECTIONS:
                    cv2.line(
                        frame,
                        points[start],
                        points[end],
                        (70, 220, 110),
                        2,
                        cv2.LINE_AA
                    )

                # Draw landmarks
                for i, point in enumerate(points):

                    color = (
                        (0, 180, 255) if i == 4 else
                        (255, 100, 80) if i == 8 else
                        (245, 245, 245)
                    )

                    cv2.circle(
                        frame,
                        point,
                        5,
                        color,
                        -1,
                        cv2.LINE_AA
                    )

                handedness = "Hand"

                if (
                    hand_index < len(result.handedness)
                    and result.handedness[hand_index]
                ):
                    handedness = (
                        result.handedness[hand_index][0]
                        .category_name
                    )

                gesture = classify_gesture(landmarks)

                hand_key = f"{handedness}_{hand_index}"
                active_keys.add(hand_key)

                if hand_key not in stable_states:
                    stable_states[hand_key] = {
                        "candidate": gesture,
                        "frames": 1,
                        "current": None,
                    }

                state = stable_states[hand_key]

                if state["candidate"] == gesture:
                    state["frames"] += 1
                else:
                    state["candidate"] = gesture
                    state["frames"] = 1

                if state["frames"] >= STABLE_FRAMES:

                    if state["current"] != gesture:

                        if gesture in GESTURES:
                            counts[gesture] += 1

                        state["current"] = gesture

                display_gesture = (
                    state["current"] or "DETECTING..."
                )

                # Gesture label near the wrist
                label_y = max(95, points[0][1] - 20)

                draw_clean_text(
                    frame,
                    f"{handedness}: {display_gesture}",
                    (points[0][0] + 12, label_y),
                    (0, 255, 255),
                    0.65
                )

                # Highlight thumb-index connection
                cv2.line(
                    frame,
                    points[4],
                    points[8],
                    (0, 255, 255)
                    if gesture == "PINCH"
                    else (220, 220, 220),
                    3,
                    cv2.LINE_AA
                )

            # Remove states for hands no longer detected
            for key in list(stable_states):
                if key not in active_keys:
                    del stable_states[key]

            # Transparent gesture counter on RIGHT SIDE
            panel_x = max(15, width - 290)

            draw_clean_text(
                frame,
                "GESTURE COUNTERS",
                (panel_x, 125),
                (80, 240, 130),
                0.65
            )

            for i, name in enumerate(GESTURES):

                y = 170 + i * 43

                draw_clean_text(
                    frame,
                    f"{name}: {counts[name]}",
                    (panel_x, y),
                    (255, 255, 255),
                    0.58
                )

            # Bottom instruction
            draw_clean_text(
                frame,
                "Show gestures clearly | Q = quit",
                (18, height - 18),
                (255, 255, 255),
                0.55
            )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if cv2.getWindowProperty(
                window_name, cv2.WND_PROP_VISIBLE
            ) < 1:
                break

    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
