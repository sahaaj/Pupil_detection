import numpy as np
import cv2
import time
haar_face=cv2.CascadeClassifier('/Users/sahaaj/Downloads/C2C/haarcascade_frontalface_default.xml')
haar_eye=cv2.CascadeClassifier("/Users/sahaaj/Downloads/C2C/haarcascade_eye.xml")
#face = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
f=cv2.imread('/Users/sahaaj/Downloads/pics/photo-1507003211169-0a1dd7228f2d.jpg')
#f=cv2.resize(f,(600,700))
'''for (x,y,w,h) in eyes:
    #cv2.rectangle(f,(x,y),(x+w,y+h),(0,0,255))
    ROI=f[y:y+h,x:x+w]
    cv2.imshow("ROI",ROI)
    ROI_gray=cv2.cvtColor(ROI,cv2.COLOR_BGR2GRAY)
    threshold=35
    kernel = np.ones((3, 3), np.uint8)
    new_frame = cv2.bilateralFilter(ROI_gray, 10, 15, 15)
    new_frame = cv2.erode(new_frame, kernel, iterations=3)
    new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]
    c = cv2.findContours(new_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    cv2.drawContours(ROI,c,-1,(255,0,0),2)
    cv2.imshow('new Frame',new_frame)
    cv2.imshow('Recording', f)
    cv2.imshow('ROI',ROI)
    k=cv2.waitKey(0)
    cv2.destroyAllWindows()
'''
def detect_faces(img):
    faces=haar_face.detectMultiScale(img)
    lista=[]
    for (x,y,w,h) in faces:
        roi=f[y:y+h,x:x+w]
        a=detect_eyes(roi)
        if(a==1):
            lista.append([x,y,w,h])
    return(lista)
def detect_eyes(img):
    eye=haar_eye.detectMultiScale(img)
    flag=0
    for (x,y,w,h) in eye:
        flag+=1
    if flag==2:
        return(1)
    return(-1)
cap=cv2.VideoCapture(0)
while True:
    _,f=cap.read()
    lista=detect_faces(f)
    for(x1,y1,w1,h1) in lista:
        roi=f[y1:y1+h1,x1:x1+w1]
        listb=haar_eye.detectMultiScale(roi)
        for (x,y,w,h) in listb:
            ROI=roi[y:y+h,x:x+w]
            #cv2.imshow("ROI",ROI)
            ROI_gray=cv2.cvtColor(ROI,cv2.COLOR_BGR2GRAY)
            threshold=35
            kernel = np.ones((3, 3), np.uint8)
            new_frame = cv2.bilateralFilter(ROI_gray, 10, 15, 15)
            new_frame = cv2.erode(new_frame, kernel, iterations=4)
            new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]
            c = cv2.findContours(new_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
            cv2.drawContours(ROI,c,-1,(255,0,0),2)
            #cv2.imshow('new Frame',new_frame)
            #cv2.imshow('Recording', f)
            #cv2.imshow('ROI',ROI)
    cv2.imshow('Recording',f)
    k=cv2.waitKey(1)
    if k==27:
        break
cv2.destroyAllWindows()