import cv2
import os
from keras.models import load_model
#from keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras.utils import img_to_array
import serial
import time

flag=0
serialcomm = serial.Serial('/dev/ttyUSB0', 9600)
serialcomm.timeout = 1

#load files
stop_sign = cv2.CascadeClassifier('cascade_stop_sign.xml')
light_classifier = cv2.CascadeClassifier('./haar_xml_07_19.xml')
classifier =load_model('./detector.model')

class_labels = ['green','red']

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
#frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
#frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

count = 0

while True:
    pwd=os.getcwd()
    #global flag
    ret, frame = cam.read()
    frame = cv2.resize(frame, (640, 480))
    
    x0=0
    y0=300
    w0=640
    h0=180
    ROAD = frame[y0:y0+h0, x0:x0+w0]
    #cv2.imshow('road', ROAD)
    
    gray = cv2.cvtColor(ROAD, cv2.COLOR_BGR2GRAY)
    image = cv2.blur(gray, (3,3)) 
    otsu_threshold, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    mask=cv2.drawContours(image, contours, -1, (255, 0, 0), 2)

    print(flag)
    if flag>0:
        a = 's#'
        serialcomm.write(a.encode())
    else:
        if len(contours) > 0 :
            c = max(contours, key=cv2.contourArea)
            cv2.drawContours(ROAD, c, -1, (0,255,0), 1)
            M = cv2.moments(c)
            if M["m00"] !=0 :
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                print("CX : "+str(cx)+"  CY : "+str(cy))
                if cx >= 320 :
                    print("Turn Left")
                    a = 'r#'
                    serialcomm.write(a.encode())
                    
                    
                if cx > 220 and cx < 320 :
                    print("On Track!")
                    a = 'f#'
                    serialcomm.write(a.encode())
                    
                    
                if cx <=220 :
                    print("Turn Right")
                    a = 'l#'
                    serialcomm.write(a.encode())
                    
                cv2.circle(ROAD, (cx,cy), 5, (255,255,255), -1)
        else :
            print("I don't see the line")
        
    
    cv2.imshow("Mask",mask)
    cv2.imshow("Frame",ROAD)
    
    x1=0
    y1=0
    w1=640
    h1=300
    SIGN = frame[y1:y1+h1, x1:x1+w1]
    cv2.imshow('sign', SIGN)
    
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
                
        
    # Display the captured frame
    cv2.imshow('Camera', frame)
    
    if cv2.waitKey(1) == ord('c'):
        print("image captured")
        cv2.imwrite("dataset/img." + '.' + str(count) + ".jpg", SIGN)
        count += 1
    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()
