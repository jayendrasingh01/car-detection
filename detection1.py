import cv2
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture('video.mp4')

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Constants
min_width_react = 80
min_height_react = 80
count_line_position = 550

# Initialize Background Subtractor
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

def center_handle(x, y, w, h):
    return int(x + w / 2), int(y + h / 2)

detect = []
offset = 6
counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)

    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)
    
    new_detect = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= min_width_react and h >= min_height_react:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center = center_handle(x, y, w, h)
            new_detect.append(center)
            cv2.circle(frame, center, 4, (0, 0, 255), -1)
    
    for (x, y) in detect[:]:
        if y < (count_line_position + offset):
            counter += 1
            detect.remove((x, y))
    
    detect = [center for center in new_detect if center not in detect]
    
    cv2.line(frame, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)
    cv2.putText(frame, f"Vehicle Counter: {counter}", (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    
    cv2.imshow('Video Original', frame)
    
    if cv2.waitKey(1) == 13:  # Enter key
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
