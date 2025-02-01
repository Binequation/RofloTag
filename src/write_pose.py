import cv2
import torch
from ultralytics import YOLO

# Загружаем модель YOLO-Pose
model = YOLO("../models/yolo11n-pose.pt")

# Путь к входному видео и выходному файлу
input_video_path = "/home/tautropy-ai/Downloads/large.mp4"  # Путь к входному видео
output_video_path = "/home/tautropy-ai/Downloads/output.mp4"  # Путь для сохранения обработанного видео

# Открываем входное видео
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Не удалось открыть видеофайл")
    exit()

# Получаем параметры видео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Создаем VideoWriter для записи обработанного видео
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек (MP4)
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

CONFIDENCE_THRESHOLD = 0.5  # Порог уверенности

cv2.namedWindow("YOLO-Pose", cv2.WINDOW_NORMAL)  # Настройка окна

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Применяем модель YOLO-Pose к текущему кадру
    results = model(frame)

    for result in results:
        # Проверяем, что keypoints существует
        if result.keypoints is not None:
            keypoints_data = result.keypoints  # Данные ключевых точек

            # Если keypoints является тензором PyTorch, преобразуем его в NumPy
            if hasattr(keypoints_data, 'xy') and hasattr(keypoints_data, 'conf'):
                try:
                    keypoints = keypoints_data.xy.cpu().numpy()  # Координаты ключевых точек
                    conf = keypoints_data.conf.cpu().numpy()  # Уверенность в ключевых точках
                except AttributeError:
                    print("Keypoints data is not in expected format. Skipping this frame.")
                    continue
            else:
                print("Keypoints do not have expected attributes. Skipping this frame.")
                continue

            for kp, cf in zip(keypoints, conf):
                if len(kp) != 17:  # Проверяем, что количество точек равно 17
                    continue

                # Формируем список валидных точек
                valid_kp = [
                    (int(x), int(y)) if c > CONFIDENCE_THRESHOLD else None
                    for (x, y), c in zip(kp, cf)
                ]

                # Рисуем ключевые точки
                for point in valid_kp:
                    if point:
                        cv2.circle(frame, point, 5, (0, 255, 0), -1)

                # Определяем скелетное соединение
                skeleton = [
                    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12),
                    (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                    (0, 1), (1, 2), (2, 3), (3, 4)
                ]

                # Соединяем ключевые точки линиями
                for (i, j) in skeleton:
                    if valid_kp[i] and valid_kp[j]:
                        cv2.line(frame, valid_kp[i], valid_kp[j], (255, 0, 0), 2)

    # Записываем обработанный кадр в выходной файл
    out.write(frame)

    # Отображаем кадр в реальном времени
    cv2.imshow("YOLO-Pose", frame)

    # Выход из цикла по кнопке 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Обработанное видео сохранено в {output_video_path}")