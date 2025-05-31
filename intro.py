import cv2
img=cv2.imread("D:\INTERNSHIP\DEEP LEARNING\opencv\data\WhatsApp Image 2025-05-24 at 9.25.04 AM.jpeg")
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
cv2.imshow('image',img)
cv2.waitKey(5000)
cv2.destroyAllWindows()