import cv2
import numpy as np
from ultralytics import YOLO

# Загружаем модель YOLO11n-pose
model = YOLO("../models/yolo11n-pose.pt")

# Открываем камеру
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("YOLO-HitboxWithID", cv2.WINDOW_NORMAL)

persons_list = []  # Список центров людей в кадре

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    new_persons = []  # Новый список людей

    for result in results:
        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)  # Hitbox координаты
            center = ((x1 + x2) // 2, (y1 + y2) // 2)  # Центр bbox'а

            # Проверяем, есть ли уже этот человек в списке
            found_index = None
            for idx, prev_center in enumerate(persons_list):
                if np.linalg.norm(np.array(center) - np.array(prev_center)) < 50:
                    found_index = idx
                    break

            if found_index is None:
                new_persons.append(center)  # Новый человек
            else:
                new_persons.append(persons_list[found_index])  # Уже существующий

    # Обновляем список людей, сохраняя порядок
    persons_list = new_persons

    # Отображаем ID и hitbox'ы
    for idx, (x1, y1, x2, y2) in enumerate(result.boxes.xyxy.cpu().numpy()):
        center = ((x1 + x2) // 2, (y1 + y2) // 2)  # Центр bbox'а

        # Отображаем прямоугольник (hitbox)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Отображаем ID внутри hitbox'а
        cv2.putText(frame, f"ID: {idx+1}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Отображаем кадр
    cv2.imshow("YOLO-Pose", frame)

    if cv2.waitKey(1) & 255 == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
