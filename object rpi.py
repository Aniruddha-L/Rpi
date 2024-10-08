import cv2
import numpy as np
from ultralytics import YOLO
import time
import RPi.GPIO as GPIO

# Load YOLOv8 model (assuming the pre-trained model is used)
model = YOLO("yolov8n.pt")  # YOLOv8 Nano version for speed

# Servo setup
SERVO_PIN = 17  # GPIO pin for the servo motor
GPIO.setmode(GPIO.BCM)  # Use BCM numbering
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)  # PWM at 50Hz
servo.start(7.5)  # Neutral position (90 degrees)

# Function to calculate the angle required to turn the camera
def calculate_turn_angle(object_x, frame_width, fov=60):
    # Camera's horizontal field of view (FoV) is assumed to be 60 degrees by default
    # object_x is the center x-coordinate of the object
    # frame_width is the width of the webcam feed
    # Calculate the offset from the center
    offset = object_x - (frame_width / 2)
    
    # Normalize the offset to [-1, 1]
    normalized_offset = offset / (frame_width / 2)
    
    # Calculate the angle to turn based on the normalized offset and the FoV
    turn_angle = normalized_offset * (fov / 2)
    
    return turn_angle

# Function to convert the calculated turn angle into a PWM duty cycle for the servo
def angle_to_duty_cycle(angle, min_angle=-90, max_angle=90, min_duty=2.5, max_duty=12.5):
    # Map the angle (-90 to 90) to a duty cycle (2.5% to 12.5%)
    duty_cycle = ((angle - min_angle) * (max_duty - min_duty) / (max_angle - min_angle)) + min_duty
    return duty_cycle

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Get the width and height of the frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set an anchor point at the center
anchor_x = frame_width // 2
anchor_y = frame_height // 2

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Perform object detection with YOLOv8
    results = model(frame)  # This returns a list of results

    # Extract bounding boxes from results
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Get xyxy format for bounding boxes
    
    if len(boxes) > 0:
        # Choose the first detected object
        box = boxes[0]
        x_min, y_min, x_max, y_max = box[:4]  # Get bounding box coordinates
        
        # Calculate center of the bounding box
        object_x = int((x_min + x_max) / 2)
        object_y = int((y_min + y_max) / 2)
        
        # Draw a rectangle around the detected object
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        
        # Draw the anchor point (center of the frame)
        cv2.circle(frame, (anchor_x, anchor_y), 5, (255, 0, 0), -1)
        
        # Draw the center of the detected object
        cv2.circle(frame, (object_x, object_y), 5, (0, 0, 255), -1)
        
        # Calculate the turn angle
        turn_angle = calculate_turn_angle(object_x, frame_width)
        
        # Print the turn angle
        print(f"Turn the camera by {turn_angle:.2f} degrees to center the object.")
        
        # Convert the turn angle to the corresponding servo duty cycle
        servo_duty_cycle = angle_to_duty_cycle(turn_angle)
        servo.ChangeDutyCycle(servo_duty_cycle)  # Send the signal to the servo
        
        # Sleep for 2 seconds before issuing another turn command
        time.sleep(2)
    
    # Display the resulting frame
    cv2.imshow('Webcam Feed', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()

# Stop the PWM and cleanup GPIO
servo.stop()
GPIO.cleanup()
