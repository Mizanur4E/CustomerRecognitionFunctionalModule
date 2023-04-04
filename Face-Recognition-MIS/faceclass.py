import warnings
warnings.filterwarnings("ignore")
import cv2
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from Retinaface_gpu.Retina_gpu import RetinaGPU
from Arcface_custom import ArcFaceRecog
from utils.face_utils import *
import time


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class FaceClass:
    def __init__(self, threshold=0.7):
        arcface_model_path = './models/model.tflite'
        self.face_rec = ArcFaceRecog.ArcFace(arcface_model_path)
        self.threshold = threshold
        print('Recognition Threshold : ', threshold)
        face_db = pd.read_csv('MIS_Face_DB.csv')
        self.known_embeddings = face_db.iloc[:, 2:]
        known_labels = face_db.iloc[:, 0]
        self.known_labels = np.array(known_labels)
        self.names = face_db.iloc[:, 1].unique()

        if torch.cuda.is_available():
            self.retina = RetinaGPU(0)  # (-1) for CPU, (0) for GPU
            print('Cuda Available, Running on GPU...')
            print('Device Name : ', torch.cuda.get_device_name(0))
        else:
            self.retina = RetinaGPU(-1)
            print('Cuda Not Found, Running on CPU...')

    def IsInside(self, ref, face):

        if ref[0] < face[0] and ref[2] > face[2] and ref[1] < face[1] and ref[3] > face[3]:
            return True
        else:
            return False

    def face_recognition(self, image, scale):

        t1 = time.time()
        faces, face_region = self.retina.extract_faces(image, scale, alignment=True)
        face_region = np.array(face_region)
        org_img = image.copy()

        tar_face = None
        tar_region = [0,0,0,0]
        name = ""
        resize_to = 1280
        img_resized = cv2.resize(org_img, (resize_to, resize_to))
        image_out = img_resized.copy()

        face_region_resized = []
        for region in face_region:
            resized_region = [region[0]*(resize_to/org_img.shape[1]), region[1]*(resize_to/org_img.shape[0]), region[2]*(resize_to/org_img.shape[1]), region[3]*(resize_to/org_img.shape[0])]
            face_region_resized.append(resized_region)

            print(region, resized_region, org_img.shape)
        face_region_resized = np.array(face_region_resized)



        ref = [400, 240 , 1000, 900]
        #cv2.rectangle(img_resized, (400,240), (1000,900), (255,0,0), 5)
        cv2.rectangle(img_resized, (840, 280), (1050, 630), (255, 0, 0), 5)
        for face, region in zip(faces, face_region_resized):
            #print(region, ref)
            if self.IsInside(ref, region):
                print('yes here')
                tar_face = face
                tar_region = region
            else:
                pass


        if tar_face is None:
            return img_resized, None, None
        else:
            try:
                face = cv2.resize(tar_face, (112, 112))
                embd = self.face_rec.calc_emb(face)
                embd = np.array(embd)
                probabilities = np.dot(self.known_embeddings, embd.T)
                index = np.argmax(probabilities)
                score = np.max(probabilities)

                if score >= self.threshold:
                    name_id = int(self.known_labels[index])
                    name = self.names[name_id]
                else:
                    name = 'unknown'

                current_time = datetime.now().strftime("%H:%M:%S")
                # if name != 'unknown':
                #     print('    Name :    {}     Score :    {}    Time: {}'.format(name, round(score, 3), current_time))
            except:
                print('Face Loading error, shape:', face.shape)

            detected_name = np.array(name)



            # print(face_region, face_region_resized, img_resized.shape)

            image_out = draw_boxes(img_resized, face_region_resized, detected_name, scale)

            t2 = time.time()
            fps = round(1 / (t2 - t1), 1)
            cv2.rectangle(image_out, (40, 0), (300, 70), (0, 255, 0), -1)
            cv2.putText(image_out, f'FPS : {fps}', (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 128, 0),
                        thickness=2, lineType=1)

            return image_out

    def face_recognition_v2(self, image, scale, memory, same_unkwn_detected ):

        t1 = time.time()
        faces, face_region = self.retina.extract_faces(image, scale, alignment=True)
        face_region = np.array(face_region)
        org_img = image.copy()
        tar_face = None



        ref = [250, 75, 450, 350]

        for face, region in zip(faces, face_region):
            print(region)
            if self.IsInside(ref, region):
                print('yes here')
                tar_face = face
                tar_region = region
            else:
                pass


        names = []

        if tar_face is None:
            return org_img
        else:

            face = cv2.resize(tar_face, (112, 112))
            embd = self.face_rec.calc_emb(face)
            embd = np.array(embd)
            memory.append(embd)
            probabilities = np.dot(self.known_embeddings, embd.T)
            index = np.argmax(probabilities)
            score = np.max(probabilities)

            if score >= self.threshold:
                name_id = int(self.known_labels[index])
                name = self.names[name_id]
                same_unkwn_detected.clear()
            else:
                name = 'unknown'
                print('unknown face', len(memory), len(same_unkwn_detected))
                if len(memory) > 1:
                    sim = np.dot(memory[-1], memory[-2].T)
                    print(sim)
                    if sim > 0.65:
                        same_unkwn_detected.append(2)
                    else:
                        same_unkwn_detected.clear()

            if len(same_unkwn_detected) > 10:
                print('New Person Detected')
                same_unkwn_detected.clear()
                for emb in memory[-10:]:
                    print(emb)


            current_time = datetime.now().strftime("%H:%M:%S")
            # except:
            #     print('Face Loading error, shape:', face.shape)

            detected_name = name
            print(same_unkwn_detected)

            # print(face_region, face_region_resized, img_resized.shape)

            image_out = draw_boxes(org_img, tar_region, detected_name, scale)

            t2 = time.time()
            fps = round(1 / (t2 - t1), 1)
            cv2.rectangle(image_out, (40, 0), (300, 70), (0, 255, 0), -1)
            cv2.putText(image_out, f'FPS : {fps}', (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 128, 0),
                        thickness=2, lineType=1)
            return org_img










        # t1 = time.time()
        # faces, face_region = self.retina.extract_faces(image, scale, alignment=True)
        # face_region = np.array(face_region)
        # org_img = image.copy()
        # detected_names = []
        # for face in faces:
        #     try:
        #         face = cv2.resize(face, (112, 112))
        #         embd = self.face_rec.calc_emb(face)
        #         embd = np.array(embd)
        #         probabilities = np.dot(self.known_embeddings, embd.T)
        #         index = np.argmax(probabilities)
        #         score = np.max(probabilities)
        #
        #         if score >= self.threshold:
        #             name_id = int(self.known_labels[index])
        #             name = self.names[name_id]
        #         else:
        #             name = 'unknown'
        #         detected_names.append(name)
        #         current_time = datetime.now().strftime("%H:%M:%S")
        #         # if name != 'unknown':
        #         #     print('    Name :    {}     Score :    {}    Time: {}'.format(name, round(score, 3), current_time))
        #     except:
        #         print('Face Loading error, shape:', face.shape)
        #
        # detected_names = np.array(detected_names)
        # resize_to = 1280
        # img_resized = cv2.resize(org_img, (resize_to, resize_to))
        # image_out = img_resized.copy()
        # face_region_resized = []
        # for region in face_region:
        #     resized_region = [region[0]*(resize_to/org_img.shape[1]), region[1]*(resize_to/org_img.shape[0]), region[2]*(resize_to/org_img.shape[1]), region[3]*(resize_to/org_img.shape[0])]
        #     face_region_resized.append(resized_region)
        # face_region_resized = np.array(face_region_resized)
        #
        # for i in range(len(face_region)):
        #     image_out = draw_boxes(img_resized, face_region_resized[i], detected_names[i], scale)
        #
        # t2 = time.time()
        # fps = round(1 / (t2 - t1), 1)
        # cv2.rectangle(image_out, (40, 0), (300, 70), (0, 0, 0), -1)
        # cv2.putText(image_out, f'FPS : {fps}', (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 128, 0), thickness=2, lineType=1)
        # return image_out, detected_names

