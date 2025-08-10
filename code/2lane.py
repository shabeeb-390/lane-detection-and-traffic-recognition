import numpy as np
import cv2
import serial
import time
from tensorflow.keras.utils import img_to_array
import os
from keras.models import load_model

#serialcomm = serial.Serial('/dev/ttyUSB0', 9600)
#serialcomm.timeout = 1

flag=0

#load files
stop_sign = cv2.CascadeClassifier('cascade_stop_sign.xml')
light_classifier = cv2.CascadeClassifier('./haar_xml_07_19.xml')
classifier =load_model('./detector.model')

class_labels = ['green','red']

x0=0
y0=300
w0=640
h0=180


def pixel_points(y1, y2, line):
    if line is None or np.isinf(line[0]) or np.isnan(line[0]):
        return None  # Ignore invalid lines
    slope, intercept = line
    x1 = int((y1 - intercept) / slope) if slope != 0 else 0  # Avoid division by zero
    x2 = int((y2 - intercept) / slope) if slope != 0 else 0
    return ((x1, int(y1)), (x2, int(y2)))


def region_selection(image):
    mask = np.zeros_like(image)
    ignore_mask_color = 255
    rows, cols = image.shape[:2]
    bottom_left  = [0, rows]
    top_left     = [0, 300]
    bottom_right = [cols, rows]
    top_right    = [640, 300]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    #cv2.imshow("mask", mask)
    return cv2.bitwise_and(image, mask)

def hough_transform(image):
    return cv2.HoughLinesP(image, 1, np.pi/180, 20, minLineLength=20, maxLineGap=500)

def average_slope_intercept(lines):
    left_lines, right_lines = [], []
    left_weights, right_weights = [], []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:
                right_lines.append((slope, intercept))
                right_weights.append(length)

    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if left_weights else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if right_weights else None
    return left_lane, right_lane

def lane_lines(image, lines):
    global flag
    print(flag)
    left_lane, right_lane = average_slope_intercept(lines)
    y1, y2 = image.shape[0], int(image.shape[0] * 0.6)
    left_line, right_line = pixel_points(y1, y2, left_lane), pixel_points(y1, y2, right_lane)

    if left_line and right_line:
        left_x, right_x = left_line[0][0], right_line[0][0]
        midpoint, image_center = (left_x + right_x) // 2, image.shape[1] // 2
        
        if flag>0:
            i = 's#'
            #serialcomm.write(i.encode())
        else:
            if abs(midpoint - image_center) <= 50:
                position = "Center"
                i="f#"
                #serialcomm.write(i.encode())
            elif midpoint < image_center:
                position = "Left"
                i="l#"
                #serialcomm.write(i.encode())
            else:
                position = "Right"
                i="r#"
                #serialcomm.write(i.encode())
            
            print(f"Vehicle Position: {position}")

    return left_line, right_line

def draw_lane_lines(image, lines, color=[0, 255, 0], thickness=5):
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

def process_frame(frame):
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayscale, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    region = region_selection(edges)
    lines = hough_transform(region)
    if lines is not None:
        return draw_lane_lines(frame, lane_lines(frame, lines))
    return frame

cap = cv2.VideoCapture(0)  # 0 for webcam, replace with 'video.mp4' for a file

while cap.isOpened():
    ret, frame = cap.read()
    print(frame.shape)
    frame = cv2.resize(frame, (640, 480))
    if not ret:
        break

    
    
    x1=0
    y1=0
    w1=636
    h1=280
    SIGN = frame[y1:y1+h1, x1:x1+w1]
    
    
    labels = []
    gray = cv2.cvtColor(SIGN,cv2.COLOR_BGR2GRAY)
    #gray = cv2.resize(gray, (224, 224))
    stop_sign_scaled = stop_sign.detectMultiScale(gray, 1.3, 5)
    light = light_classifier.detectMultiScale(gray,1.3,5)

    if len(stop_sign_scaled)==0:
        flag=0
    else:
        flag=1
    
    # Detect the stop sign, x,y = origin points, w = width, h = height
    for (x1, y1, w1, h1) in stop_sign_scaled:
        # Draw rectangle around the stop sign
        stop_sign_rectangle = cv2.rectangle(frame, (x1,y1),
                                            (x1+w1, y1+h1),
                                            (0, 255, 0), 3)
        # Write "Stop sign" on the bottom of the rectangle
        stop_sign_text = cv2.putText(img=stop_sign_rectangle,
                                     text="Stop Sign",
                                     org=(x1, y1+h1+30),
                                     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                     fontScale=1, color=(0, 0, 255),
                                     thickness=2, lineType=cv2.LINE_4)

    for (x,y,w,h) in light:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        #roi = gray[y:y+h,x:x+w]
        
        #width,height = roi.shape
        midx = x+int(w/2)
        midy = y+int(h/2)
        pt1 = (midx,y+int(0.2*h))
        pt2 = (midx,y+int(0.8*h))
        
        print(midx,midy, pt1, pt2)
        #cv2.line(frame, (midx,y+int(0.2*h)), (midx,y+int(0.8*h)), (255,0,0),2)
        
        maxvalue = 0
        maxy = 0
        for k in range (y+(int(0.2*h)), (y+int(0.8*h)), 1):
            value = gray[k,midx]
            print(value, midx, k)
            if value > maxvalue:
                maxvalue = value
                maxy = k
        print("max", maxvalue, maxy, midy)       
        if maxvalue > 200:
            if maxy > midy:
                cv2.putText(frame,"green",(midx, maxy), cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
                print("green")
                flag=0
            else:
                cv2.putText(frame,"red",(midx, maxy), cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
                print("red")
                flag=2
                
    processed_frame = process_frame(frame)
    cv2.imshow("Lane Detection", processed_frame)
    
    cv2.imshow('sign', SIGN)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#serialcomm.close()

