import time
import cv2
import threading
import pyttsx3
from gtts import gTTS
from pygame import mixer
from utils.videocapture import CustomVideoCapture
from faceclass import FaceClass


class ReceptionBot:
    def __init__(self):
        self.url = 'rtsp://admin:AiBi@8899@192.168.101.80:554'
        # self.url = 0
        self.engine = pyttsx3.init()
        mixer.init()
        self.FC = FaceClass(threshold=0.7)
        self.scale = 0.3
        self.text = ''
        # threading.Thread(target=self.text_to_speech, daemon=True).start()
        threading.Thread(target=self.make_sound, daemon=True).start()

    def text_to_speech(self):
        while True:
            if self.text != '':
                print(self.text)
                self.engine.say(f'{self.text}')
                time.sleep(1)
            self.engine.runAndWait()
            self.engine.stop()

    def make_sound(self):
        while True:
            if self.text != '':
                myobj = gTTS(self.text, lang='en', slow=True)
                myobj.save('./audio_data/welcome'+'.mp3')
                mixer.music.load('./audio_data/welcome'+'.mp3')
                mixer.music.play(0)
                while(mixer.music.get_busy() == True):
                    continue
            mixer.music.stop()

    def process_video(self):
        cap = CustomVideoCapture(self.url)
        if cap.isOpened():
            while True:
                frame = cap.read()
                image, detected_names = self.FC.face_recognition(frame, self.scale)
                self.text = ''
                for name in detected_names:
                    if name != 'unknown':
                        self.text = self.text + 'Mr.' + name + '\n'
                if self.text != '':
                    self.text = 'welcome ' + '\n' + self.text + '\n' + 'How can i help you?'
                image = cv2.resize(image, (1500, 900))
                cv2.imshow('Face Recognition MIS', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()
        else:
            print('Unable to connect camera')


if __name__ == '__main__':
    ReceptionBot().process_video()

