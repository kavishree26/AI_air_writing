
import cv2
import mediapipe as mp
import numpy as np
import os

# Find model file
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hand_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "hand_landmarker.task not found! "
        "Download it into the project folder."
    )

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.65,
    min_hand_presence_confidence=0.65,
    min_tracking_confidence=0.65
)

# Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

canvas = None
previous_point = None
smooth_point = None

brush_color = (0, 255, 0)
brush_size = 6

# Finger state detection
def is_finger_up(landmarks, tip, pip):
    return landmarks[tip].y < landmarks[pip].y

try:
    with HandLandmarker.create_from_options(options) as detector:

        while True:
            success, frame = cap.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)

            height, width, _ = frame.shape

            if canvas is None:
                canvas = np.zeros_like(frame)

            # Convert to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = int(
                cv2.getTickCount() * 1000 /
                cv2.getTickFrequency()
            )

            result = detector.detect_for_video(
                mp_image, timestamp
            )

            drawing = False

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                # Index finger
                index_up = is_finger_up(hand, 8, 6)

                # Other fingers
                middle_up = is_finger_up(hand, 12, 10)
                ring_up = is_finger_up(hand, 16, 14)
                pinky_up = is_finger_up(hand, 20, 18)

                # Only index finger raised
                drawing = (
                    index_up
                    and not middle_up
                    and not ring_up
                    and not pinky_up
                )

                # Index fingertip position
                x = int(hand[8].x * width)
                y = int(hand[8].y * height)

                # Draw hand landmarks
                for point in hand:
                    px = int(point.x * width)
                    py = int(point.y * height)

                    cv2.circle(
                        frame,
                        (px, py),
                        3,
                        (255, 150, 0),
                        -1
                    )

                # Smooth finger movement
                if smooth_point is None:
                    smooth_point = (x, y)
                else:
                    alpha = 0.65
                    smooth_point = (
                        int(alpha * x +
                            (1 - alpha) * smooth_point[0]),
                        int(alpha * y +
                            (1 - alpha) * smooth_point[1])
                    )

                x, y = smooth_point

                cv2.circle(
                    frame,
                    (x, y),
                    10,
                    (0, 0, 255),
                    -1
                )

                if drawing:
                    if previous_point is not None:
                        cv2.line(
                            canvas,
                            previous_point,
                            (x, y),
                            brush_color,
                            brush_size
                        )

                    previous_point = (x, y)

                else:
                    previous_point = None

            else:
                previous_point = None
                smooth_point = None

            # Combine camera and drawing
            output = cv2.add(frame, canvas)

            status = "DRAWING" if drawing else "NOT DRAWING"

            cv2.putText(
                output,
                "AIR WRITING",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            cv2.putText(
                output,
                "Status: " + status,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if drawing else (0, 0, 255),
                2
            )

            cv2.putText(
                output,
                "Index Up: Draw | C: Clear | Q: Quit",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.imshow("Air Writing - Hand Tracking", output)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                canvas = np.zeros_like(frame)
                previous_point = None

            elif key == ord('q'):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()


