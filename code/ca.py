import cv2
import os

# Open the default camera
cam = cv2.VideoCapture(2)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

count = 0

while True:
    pwd=os.getcwd()
    
    ret, frame = cam.read()
    
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
                a = 'l#'
                #serialcomm.write(a.encode())
                
                
            if cx > 220 and cx < 320 :
                print("On Track!")
                a = 'f#'
                #serialcomm.write(a.encode())
                
                
            if cx <=220 :
                print("Turn Right")
                a = 'r#'
                #serialcomm.write(a.encode())
                
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
