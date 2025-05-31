import cv2
video = cv2.VideoCapture('D:\INTERNSHIP\DEEP LEARNING\opencv\data\WhatsApp Video 2025-05-27 at 9.35.50 AM.mp4')

while True:
    success,frame =video.read()
    print(success)
    if success == True:
        grayVideo = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        thresh, bw_img = cv2.threshold(grayVideo,128,255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)
    else:
        print('video reading fail')
    cv2.imshow('video',bw_img)
    if cv2.waitKey(1)& 0xFF==27:
        break

video.release()
cv2.destroyAllWindows ()