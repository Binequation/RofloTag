import cv2
import torch
from ultralytics import YOLO

# Загружаем YOLO-Pose
model = YOLO("../models/yolov8n-pose.pt")

# Открываем камеру
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

CONFIDENCE_THRESHOLD = 0.5  # Порог уверенности

cv2.namedWindow("YOLO-Pose", cv2.WINDOW_NORMAL)  # Настройка окна

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()
        conf = result.keypoints.conf.cpu().numpy()

        for kp, cf in zip(keypoints, conf):
            if len(kp) != 17:
                continue

            valid_kp = [(int(x), int(y)) if c > CONFIDENCE_THRESHOLD else None for (x, y), c in zip(kp, cf)]

            for point in valid_kp:
                if point:
                    cv2.circle(frame, point, 5, (0, 255, 0), -1)

            skeleton = [
                (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12),
                (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                (0, 1), (1, 2), (2, 3), (3, 4)
            ]

            for (i, j) in skeleton:
                if valid_kp[i] and valid_kp[j]:
                    cv2.line(frame, valid_kp[i], valid_kp[j], (255, 0, 0), 2)

    # Масштабирование до 1280x720
    frame = cv2.resize(frame, (1280, 720))

    cv2.imshow("YOLO-Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()