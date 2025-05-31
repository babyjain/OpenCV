import cv2
img =cv2.imread('D:\INTERNSHIP\DEEP LEARNING\opencv\data\WhatsApp Image 2025-05-24 at 9.25.05 AM.jpeg')
face_casecode =cv2.CascadeClassifier('D:\INTERNSHIP\DEEP LEARNING\opencv\data\haarcascade_eye.xml')
face = face_casecode.detectMultiScale(img)
print(face)
for(x,y,w,h) in face : 
    cv2.rectangle(img,pt1 = (x,y),pt2 = (x+w,y+h),color=(0,255,0),thickness=2)
cv2.imshow('face detection',img)
cv2.waitKey(50000)
cv2.destroyAllWindows()

