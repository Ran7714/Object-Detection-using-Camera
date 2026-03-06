import cv2
from ultralytics import RTDETR

model = RTDETR("rtdetr-x.pt") 
def detect_objects(frame):
    results = model(frame, conf=0.3, iou=0.5)  # Lower confidence threshold
    detected_objects = []

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            confidence = box.conf[0].item()

            label = model.names[class_id]
            detected_objects.append(f"{label} ({confidence:.2f})")

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Show label + confidence score on frame
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame, detected_objects

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Higher resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, detected_objects = detect_objects(frame)
        cv2.imshow("AI Vision", frame)

        if detected_objects:
            print("Detected:", detected_objects)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press Q to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()