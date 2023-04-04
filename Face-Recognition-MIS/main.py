import cv2
import argparse
from utils.videocapture import CustomVideoCapture
from faceclass import FaceClass
from project_utils import AttendanceManagement

scale = 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=False, action="store_true")
    parser.add_argument("--server", default=False, action="store_true")
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()

    process_video_v2()

    # if args.image:
    #     process_image()
    # else:
    #     if args.server:
    #         process_video(show=False, threshold=args.threshold)
    #     else:
    #         process_video(show=True, threshold=args.threshold)


def process_image():

    image_path = './test/test2.jpg'
    frame = cv2.imread(image_path)
    FC = FaceClass(threshold=0.6)
    image= FC.face_recognition(frame, 1)
    image = cv2.resize(image, (0, 0), fx=0.6, fy=0.6)
    cv2.imshow('Face Recognition MIS', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def process_video(show=True, threshold=0.65):

    #url = 'rtsp://admin:AiBi@8899@192.168.101.80:554'
    url = 0
    # url = 'rtsp://admin:datacenternvr@192.168.100.125:554'
    cap = CustomVideoCapture(url)
    FC = FaceClass(threshold=threshold)
    if cap.isOpened():
        while True:
            frame = cap.read()
            image = FC.face_recognition(frame, scale)

            if show:

                cv2.imshow('Face Recognition MIS', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()
    else:
        print('Unable to connect camera')

def process_video_v2(show=True, threshold=0.65):


    url = 0
    cap = CustomVideoCapture(url)
    FC = FaceClass(threshold=threshold)
    memory =[]
    same_unkwn_detected = []
    if cap.isOpened():
        while True:
            frame = cap.read()
            if len(memory) > 512:
                memory = memory[-256:]
            print(len(memory))
            image = FC.face_recognition_v2(frame, scale, memory, same_unkwn_detected)
            #image = cv2.resize(image, (1080,1080))
            if show:

                cv2.rectangle(image, (250, 75), (450, 350), (255, 0, 0), 5)
                image = cv2.resize(image, (880,580))
                cv2.imshow('Face Recognition MIS', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()
    else:
        print('Unable to connect camera')




if __name__ == '__main__':
    main()

