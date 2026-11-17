"""
Vision Verse - MediaPipe Hand Tracking Test
Tracks up to two hands and displays all 21 landmarks, thumb/index tips,
and thumb-index pinch status.

Run:
    py -3.13 02_hand_tracking.py
Press Q or ESC to quit.
"""

from pathlib import Path
import math
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    BASE_DIR / "hand_landmarker.task",
    BASE_DIR / "models" / "hand_landmarker.task",
]
MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.is_file()), None)

# MediaPipe hand landmark connections (21 landmarks per hand).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def open_camera():
    """Open the default Windows webcam, with a backend fallback."""
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)
    if camera.isOpened():
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


def landmark_xy(landmark, width, height):
    x = max(0, min(width - 1, int(landmark.x * width)))
    y = max(0, min(height - 1, int(landmark.y * height)))
    return x, y


def main():
    if MODEL_PATH is None:
        expected = BASE_DIR / "hand_landmarker.task"
        raise SystemExit(
            "MediaPipe model file was not found.\n"
            f"Place hand_landmarker.task here:\n{expected}\n"
            "The Vision Verse ZIP should already include this file."
        )

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )

    detector = vision.HandLandmarker.create_from_options(options)
    camera = open_camera()

    if not camera.isOpened():
        detector.close()
        raise SystemExit(
            "Could not open webcam. Close other camera apps and check "
            "Windows Settings > Privacy & security > Camera."
        )

    print("Vision Verse hand tracking started. Press Q or ESC to quit.")
    start_time = time.perf_counter()
    last_timestamp = -1
    window_name = "Vision Verse - MediaPipe Hand Tracking"

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a webcam frame.")
                break

            # Mirror the camera for natural interaction.
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = max(
                int((time.perf_counter() - start_time) * 1000),
                last_timestamp + 1,
            )
            last_timestamp = timestamp_ms

            result = detector.detect_for_video(mp_image, timestamp_ms)

            detected_count = len(result.hand_landmarks)
            cv2.rectangle(frame, (12, 12), (330, 54), (25, 25, 25), -1)
            cv2.putText(
                frame, f"Hands detected: {detected_count}",
                (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (80, 240, 130), 2, cv2.LINE_AA
            )

            for hand_index, landmarks in enumerate(result.hand_landmarks):
                points = [landmark_xy(lm, width, height) for lm in landmarks]

                # Draw the 21-point skeleton.
                for start, end in HAND_CONNECTIONS:
                    cv2.line(
                        frame, points[start], points[end],
                        (70, 220, 110), 2, cv2.LINE_AA
                    )

                for index, point in enumerate(points):
                    radius = 6 if index in (4, 8) else 3
                    color = (0, 180, 255) if index == 4 else (
                        (255, 100, 80) if index == 8 else (245, 245, 245)
                    )
                    cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)

                thumb_tip = points[4]
                index_tip = points[8]
                wrist = points[0]
                middle_mcp = points[9]

                # Scale the pinch threshold to the size of the detected hand.
                hand_scale = max(
                    1.0,
                    (math.dist(wrist, middle_mcp) +
                     math.dist(points[5], points[17])) / 2.0
                )
                pinch_ratio = math.dist(thumb_tip, index_tip) / hand_scale
                is_pinching = pinch_ratio < 0.38

                cv2.line(
                    frame, thumb_tip, index_tip,
                    (0, 255, 255) if is_pinching else (220, 220, 220),
                    2, cv2.LINE_AA
                )

                handedness = "Hand"
                if hand_index < len(result.handedness) and result.handedness[hand_index]:
                    handedness = result.handedness[hand_index][0].category_name

                status = "PINCH" if is_pinching else "OPEN"
                label_y = max(78, points[0][1] - 18)
                cv2.putText(
                    frame, f"{handedness}: {status}",
                    (points[0][0] + 10, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 255) if is_pinching else (255, 255, 255),
                    2, cv2.LINE_AA
                )

                cv2.putText(
                    frame, "THUMB", (thumb_tip[0] + 8, thumb_tip[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 180, 255), 1,
                    cv2.LINE_AA
                )
                cv2.putText(
                    frame, "INDEX", (index_tip[0] + 8, index_tip[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 100, 80), 1,
                    cv2.LINE_AA
                )

            cv2.putText(
                frame, "Move your hand into view | Pinch thumb + index | Q = quit",
                (18, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
