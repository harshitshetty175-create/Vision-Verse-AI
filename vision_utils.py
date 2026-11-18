"""MediaPipe Tasks API helpers."""
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def create_hand_landmarker(model_path, num_hands=1):
    base = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)

def detect_hands(detector, frame_rgb, timestamp_ms):
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    return detector.detect_for_video(image, timestamp_ms)


def draw_hand(frame, hand_landmarks, color=(60, 180, 90)):
    """Draw hand landmarks and connections, returning landmark pixel points."""
    import cv2
    import mediapipe as mp

    height, width = frame.shape[:2]

    # Convert normalised landmark coordinates to pixel coordinates.
    points = [
        (int(lm.x * width), int(lm.y * height))
        for lm in hand_landmarks
    ]

    # Draw the connections between landmarks.
    connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

    for connection in connections:
        start = points[connection.start]
        end = points[connection.end]
        cv2.line(frame, start, end, color, 2)

    # Draw each landmark.
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    return points
