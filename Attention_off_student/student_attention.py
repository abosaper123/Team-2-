import cv2
import cvzone
from ultralytics import YOLO
import math

model = YOLO(r"C:\Users\TC\PycharmProjects\pythonProject14\best.pt")

cap = cv2.VideoCapture(0)

class_name = ['raising hand', 'read', 'write', 'sleep']

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (680, 480))

    result = model(frame, stream=True)

    for info in result:
        boxes = info.boxes

        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Classe = int(box.cls[0])

            if confidence > 50:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{class_name[Classe]} {confidence}%', [x1 + 8, y1 + 100], scale=1.5,
                                   thickness=2)

                if class_name[Classe] == 'sleep':
                    print("Warning: Sleep detected!")

                    cvzone.putTextRect(frame, "Sleep Detected!", [50, 50], scale=2, thickness=3, colorR=(0, 0, 255))

    cv2.imshow("Intern Team2", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
