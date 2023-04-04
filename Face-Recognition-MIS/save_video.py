import cv2

url = 'rtsp://admin:AiBi@8899@192.168.101.80:554'
# url = 0

username = input("Enter PersonName:")
print("PersonName is: " + username)

video1 = cv2.VideoCapture(url)
frame_rate = video1.get(5)
resized_to = (int(video1.get(3)), int(video1.get(4)))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out1 = cv2.VideoWriter(f'./all_videos/{username}.mp4', fourcc, frame_rate, resized_to)


while video1.isOpened():
    ret1, frame1 = video1.read()
    if ret1: 
        resized = cv2.resize(frame1,(resized_to))
        out1.write(resized)
        cv2.imshow('F', resized)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break


out1.release()
cv2.destroyAllWindows()
print('Done')
