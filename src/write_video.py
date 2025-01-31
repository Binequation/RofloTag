import cv2
import torch
from ultralytics import YOLO

# Загружаем YOLO-Pose
model = YOLO("yolov8n-pose.pt")

# Открываем видеофайл
input_video = "input.mp4"  # Путь к входному видео
output_video = "output.mp4"  # Путь к сохранённому видео
cap = cv2.VideoCapture(input_video)

# Получаем параметры видео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Частота кадров

# Настраиваем сохранение видео
fourcc = cv2.VideoWriter_fourcc(*'mp4')
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

CONFIDENCE_THRESHOLD = 0.5  # Минимальный порог уверенности

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Запускаем YOLO-Pose
    results = model(frame)

    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()  # Координаты суставов
        conf = result.keypoints.conf.cpu().numpy()  # Уверенность точек

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

    out.write(frame)  # Сохраняем кадр в файл
    cv2.imshow("YOLO-Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
