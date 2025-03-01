from ultralytics import YOLO
import cv2
import cvzone
import math

# Taking input
cap = cv2.VideoCapture(0)
model = YOLO("CustomTrainedFireModel.pt")

classNames = ["FIRE"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    results = model(frame, stream=True)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            clsid = int(box.cls[0])

            if confidence > 20:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 6)
                cvzone.putTextRect(frame, f"{classNames[clsid]} {confidence}%", [x1 + 8, y1 + 100], scale=1.5,
                                   thickness=2)

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

