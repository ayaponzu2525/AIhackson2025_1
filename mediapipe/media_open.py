import cv2
import mediapipe as mp
import math
import numpy as np
import time

# MediaPipe Pose準備
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# HaarCascade顔検出器をロード
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# カメラ起動
cap = cv2.VideoCapture(0)

# 初期化
prev_shoulder_position = None
total_movement = 0
shoulder_still_count = 0
last_check_time = time.time()

# しきい値設定
MOVEMENT_THRESHOLD = 150  # px
CHECK_INTERVAL = 10      # 10秒ごとにチェック

def calculate_neck_angle(shoulder, nose):
    """肩中心→鼻のベクトルと真上方向ベクトルとのなす角を計算"""
    vector = np.array([nose[0] - shoulder[0], nose[1] - shoulder[1]])
    vertical = np.array([0, -1])
    cos_theta = np.dot(vector, vertical) / (np.linalg.norm(vector) * np.linalg.norm(vertical))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = math.degrees(math.acos(cos_theta))
    return angle

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)

    if len(faces) > 0:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # 座標取得
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            nose = landmarks[mp_pose.PoseLandmark.NOSE]

            h, w, _ = image.shape

            left_shoulder_point = (int(left_shoulder.x * w), int(left_shoulder.y * h))
            right_shoulder_point = (int(right_shoulder.x * w), int(right_shoulder.y * h))
            nose_point = (int(nose.x * w), int(nose.y * h))

            # 肩を結ぶ線
            cv2.line(image, left_shoulder_point, right_shoulder_point, (255, 255, 0), 2)

            # 肩中心から鼻までの線
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            shoulder_center = (shoulder_center_x * w, shoulder_center_y * h)
            nose_point = (nose.x * w, nose.y * h)

            cv2.line(image, (int(shoulder_center[0]), int(shoulder_center[1])), (int(nose_point[0]), int(nose_point[1])), (0, 255, 0), 2)

            # 首角度計算
            neck_angle = calculate_neck_angle(shoulder_center, nose_point)

            # 鼻と肩中心の距離
            distance = np.linalg.norm(np.array(nose_point) - np.array(shoulder_center))

            # 距離が短すぎたら前傾
            if distance < 50:
                cv2.putText(image, 'Head Drop!', (30, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)

            # 首角度表示
            cv2.putText(image, f'Neck Tilt: {neck_angle:.1f} deg', (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            if neck_angle > 20:
                cv2.putText(image, 'Bad Posture!', (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # ----------------------------
            # ここから肩動きリアルタイム計算！！
            current_position = np.array([
                left_shoulder_point[0], left_shoulder_point[1],
                right_shoulder_point[0], right_shoulder_point[1]
            ])

            if prev_shoulder_position is not None:
                move_distance = np.linalg.norm(current_position - prev_shoulder_position)
                total_movement += move_distance

            prev_shoulder_position = current_position
            # ----------------------------

    else:
        image = frame.copy()
        cv2.putText(image, 'No Face Detected!', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # 10秒ごとにtotal_movementをチェック
    current_time = time.time()
    if current_time - last_check_time >= CHECK_INTERVAL:
        if total_movement < MOVEMENT_THRESHOLD:
            shoulder_still_count += 1
            cv2.putText(image, 'Shoulder Stillness Detected!', (30, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)

        total_movement = 0  # リセット
        last_check_time = current_time

    # ----------------------------
    # 毎フレームリアルタイム情報表示！！
    cv2.putText(image, f'Shoulder Stillness Count: {shoulder_still_count}', (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 2)
    cv2.putText(image, f'Movement: {total_movement:.1f}px', (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2)
    # ----------------------------

    cv2.imshow('Posture and Face Check', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
