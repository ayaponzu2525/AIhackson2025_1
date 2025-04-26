import cv2
import mediapipe as mp

# MediaPipe Pose準備
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# カメラ起動
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 画像をRGBに変換
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # 画像をBGRに戻す
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 骨格を描画
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('MediaPipe Pose', image)

    if cv2.waitKey(5) & 0xFF == 27:  # ESCキーで終了
        break

cap.release()
cv2.destroyAllWindows()
