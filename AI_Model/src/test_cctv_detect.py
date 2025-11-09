import cv2
from ultralytics import YOLO
import time

from ultralytics.models.yolo import model

# -- 1. Load Model --
model_path = "/home/parichu/Ai/all_workspace/service_robot_ws2/fire_smoke_model_yolo11n.pt"

# -- 1.1 Interrupt For Check Path --
try:
    model = YOLO(model_path)
    print(f"Load model {model_path} Succese")
except Exception as e:
    print(f"Error Can't Load Model {e}")
    exit()

# -- 2. Open Camera --
# If Use Web Camera(laptop) Use port = 0
# If Use Extend Camera Use port = 1, 2, ...

camera_id = 2
cap = cv2.VideoCapture(camera_id)


# -- 2.1 Check For Open Camera --
if not cap.isOpened():
    print(f"Error can't Open Camera {camera_id}")
    print("Please Check Camera Port")
    exit()

print("Start Detect From Camera (press 'q' for exit)")

# -- 2.2 While Loop --
while True:
    # -- 3. Read Frame From Camera --
    ret, frame = cap.read()

    # If ret is false (Can't reade frame) exit Loop
    if not ret:
        print("Can't Detect Frame")
        break

    # -- 4. Send Detection to Model --
    results = model(frame)

    # -- 5. Draw Result (Create Bounding Box) into frame --
    #.plot() Is Result From YOLO For Draw Result Auto
    annotated_frame = results[0].plot()

    # -- 6. Show Result --
    cv2.imshow("CCTV Simulate - AI Detection (press 'q' to exit)", annotated_frame)

    # -- 7. Set Delay for wait input for 'q' to exit --
    # cv2.waitKey(1) Delay 1ms
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -- 8. Return Resouce Camera and Close Ui --
cap.release()
cv2.destroyAllWindows()
print("--Close--")


