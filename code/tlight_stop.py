from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

stop_sign = cv2.CascadeClassifier('cascade_stop_sign.xml')
light_classifier = cv2.CascadeClassifier('./haar_xml_07_19.xml')
classifier =load_model('./detector.model')

class_labels = ['green','red']

cap = cv2.VideoCapture(0)
# cap.set(3, 1200)
# cap.set(4, 720)


while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #gray = cv2.resize(gray, (224, 224))
    stop_sign_scaled = stop_sign.detectMultiScale(gray, 1.3, 5)
    light = light_classifier.detectMultiScale(gray,1.3,5)
    
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
            else:
                cv2.putText(frame,"red",(midx, maxy), cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
                print("red")
        
    cv2.imshow('Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

























