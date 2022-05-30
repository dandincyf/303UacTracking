import cv2
idd=0
while(1):
    cap=cv2.VideoCapture(0)
    ret,frame=cap.read()
    if ret==False:
        idd+=1
    else:
        print(idd)
        break