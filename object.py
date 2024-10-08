import cv2
import numpy as np
from ultralytics import YOLO
import time

model = YOLO("yolov8n.pt")  

def calculate_turn_angle(object_x, frame_width, fov=60):
    offset = object_x - (frame_width / 2)
    
    normalized_offset = offset / (frame_width / 2)
    
    turn_angle = normalized_offset * (fov / 2)
    
    return turn_angle

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

anchor_x = frame_width // 2
anchor_y = frame_height // 2

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame) 

    boxes = results[0].boxes.xyxy.cpu().numpy()  
    
    if len(boxes) > 0:
        box = boxes[0]
        x_min, y_min, x_max, y_max = box[:4] 
        
        object_x = int((x_min + x_max) / 2)
        object_y = int((y_min + y_max) / 2)
        
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        
        cv2.circle(frame, (anchor_x, anchor_y), 5, (255, 0, 0), -1)
        
        cv2.circle(frame, (object_x, object_y), 5, (0, 0, 255), -1)
        
        turn_angle = calculate_turn_angle(object_x, frame_width)
        
        print(f"Turn the camera by {turn_angle:.2f} degrees to center the object.")

        time.sleep(2)
    
    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
