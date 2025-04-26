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

# 肩位置を保存するリスト
shoulder_positions = []
last_check_time = time.time()

# しきい値設定
MOVEMENT_THRESHOLD = 30  # 1分間でこの距離(px)未満なら疲労警告
CHECK_INTERVAL = 10      # チェック間隔（秒）
shoulder_still_count = 0  # 肩静止の検出回数


def calculate_neck_angle(shoulder, nose):
    """肩中心→鼻のベクトルと、真上ベクトル（0, -1）のなす角度を計算"""
    vector = np.array([nose[0] - shoulder[0], nose[1] - shoulder[1]])  # 肩→鼻
    vertical = np.array([0, -1])  # 真上方向
    cos_theta = np.dot(vector, vertical) / (np.linalg.norm(vector) * np.linalg.norm(vertical))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 数値誤差対策
    angle = math.degrees(math.acos(cos_theta))
    return angle

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 顔検出
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)

    if len(faces) > 0:
        # 顔がある場合 → MediaPipe処理
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # ランドマーク取得
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            nose = landmarks[mp_pose.PoseLandmark.NOSE]

            # まずh, wを取る！
            h, w, _ = image.shape

            # ここから座標変換（wとhを使う！）
            left_shoulder_point = (int(left_shoulder.x * w), int(left_shoulder.y * h))
            right_shoulder_point = (int(right_shoulder.x * w), int(right_shoulder.y * h))
            nose_point = (int(nose.x * w), int(nose.y * h))

            
            cv2.line(image, left_shoulder_point, right_shoulder_point, (255, 255, 0), 2)  # 肩の線：水色

            nose = landmarks[mp_pose.PoseLandmark.NOSE]

            # 肩の中心
            h, w, _ = image.shape
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            shoulder_center = (shoulder_center_x * w, shoulder_center_y * h)
            nose_point = (nose.x * w, nose.y * h)

            # 線を描画
            cv2.line(image, (int(shoulder_center[0]), int(shoulder_center[1])), (int(nose_point[0]), int(nose_point[1])), (0, 255, 0), 2)

            # 首の傾き角度を計算
            neck_angle = calculate_neck_angle(shoulder_center, nose_point)

            # 距離を計算（鼻と肩中心の距離）
            distance = np.linalg.norm(np.array(nose_point) - np.array(shoulder_center))

            # 肩の位置を記録
            shoulder_positions.append((left_shoulder.x, left_shoulder.y, right_shoulder.x, right_shoulder.y))

            # 距離が短すぎたらBad前傾
            if distance < 50:
                cv2.putText(image, 'Head Drop!', (30, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)

            # 首の角度表示
            cv2.putText(image, f'Neck Tilt: {neck_angle:.1f} deg', (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # 姿勢悪いか判定
            if neck_angle > 20:
                cv2.putText(image, 'Bad Posture!', (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    else:
        # 顔がない場合は警告だけ表示
        image = frame.copy()
        cv2.putText(image, 'No Face Detected!', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # 肩の動き量を一定時間ごとにチェック
    current_time = time.time()
    if current_time - last_check_time >= CHECK_INTERVAL:
        total_movement = 0
        for i in range(1, len(shoulder_positions)):
            left_move = np.linalg.norm(np.array(shoulder_positions[i][:2]) - np.array(shoulder_positions[i-1][:2]))
            right_move = np.linalg.norm(np.array(shoulder_positions[i][2:]) - np.array(shoulder_positions[i-1][2:]))
            total_movement += left_move + right_move

        if total_movement < MOVEMENT_THRESHOLD:
            shoulder_still_alert = True
            alert_start_time = time.time()
            shoulder_still_count += 1  # ★ここでカウントアップ！！

            cv2.putText(image, 'Shoulder Stillness Detected!', (30, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)

        # リセット
        shoulder_positions.clear()
        last_check_time = current_time
        
    cv2.putText(image, f'Shoulder Stillness Count: {shoulder_still_count}', (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 2)

    cv2.rectangle(image, (30, 30), (30+30, 30+30), (0, 255, 255), 2)


    cv2.imshow('Posture and Face Check', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
