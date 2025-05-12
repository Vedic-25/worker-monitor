
import cv2
import mediapipe as mp
import numpy as np

import csv
from datetime import datetime

log_file = "activity_log.csv"


with open(log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Activity"])

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

# Video input
video_path = "video_2.mp4"
cap = cv2.VideoCapture(video_path)

prev_landmarks = None  

current_state = "Unknown"
state_counter = 0
state_threshold = 10  

def calculate_angle(a, b, c):
    """Calculate angle between three points."""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def is_working_posture(landmarks, prev_landmarks=None):
    if not landmarks:
        return False

    lw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    le = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    re = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    required = [lw, rw, le, re, ls, rs]
    if not all(lm.visibility > 0.3 for lm in required):
        return False

    left_angle = calculate_angle(ls, le, lw)
    right_angle = calculate_angle(rs, re, rw)
    elbows_bent = (45 < left_angle < 160) and (45 < right_angle < 160)

    wrist_above_hip = lw.y < le.y and rw.y < re.y
    posture_based = elbows_bent or wrist_above_hip

    motion_based = False
    if prev_landmarks:
        lw_prev = prev_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        rw_prev = prev_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        motion = np.linalg.norm([lw.x - lw_prev.x, lw.y - lw_prev.y]) + \
                 np.linalg.norm([rw.x - rw_prev.x, rw.y - rw_prev.y])
        if motion > 0.01:  
            motion_based = True

    return posture_based or motion_based

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    label = current_state
    color = (100, 100, 100)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        confidences = [lm.visibility for lm in landmarks]
        if np.mean(confidences) > 0.3:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            working = is_working_posture(landmarks, prev_landmarks)
            prev_landmarks = landmarks

            predicted_state = "Working" if working else "Idle"

            if predicted_state != current_state:
                state_counter += 1
                if state_counter >= state_threshold:
                    current_state = predicted_state
                    # current_state = predicted_state
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(log_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([timestamp, current_state])
                    state_counter = 0
            else:
                state_counter = 0

            label = current_state
            color = (0, 255, 0) if current_state == "Working" else (0, 0, 255)
        else:
            label = "Low Confidence"
            color = (0, 255, 255)
            state_counter = 0
    else:
        prev_landmarks = None
        label = "No Worker Detected"
        color = (100, 100, 100)
        state_counter = 0

    cv2.putText(image, f"Status: {label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
    cv2.imshow('Worker Activity Detection', image)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
