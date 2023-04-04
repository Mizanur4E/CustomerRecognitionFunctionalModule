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
from csv import writer


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class FaceClass:
    def __init__(self, threshold=0.7):
        arcface_model_path = './models/model.tflite'
        self.face_rec = ArcFaceRecog.ArcFace(arcface_model_path)
        self.threshold = threshold
        print('Recognition Threshold : ', threshold)
        self.face_db = pd.read_csv('MIS_Face_DB.csv')
        self.known_embeddings = self.face_db.iloc[:, 2:]
        known_labels = self.face_db.iloc[:, 0]
        self.known_labels = np.array(known_labels)
        self.names = self.face_db.iloc[:, 1]

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


    def face_recognition_v2(self, image, scale, memory, same_unkwn_detected ):

        t1 = time.time()
        faces, face_region = self.retina.extract_faces(image, scale, alignment=True)
        face_region = np.array(face_region)
        org_img = image.copy()
        tar_face = None
        area = 0


        ref = [250, 75, 450, 350]
        box_area = abs((ref[3]-ref[1])*(ref[2]-ref[0]))

        for face, region in zip(faces, face_region):
            print(region)
            if self.IsInside(ref, region):
                print('yes here')

                new_area = abs((region[3]-region[1])*(region[2]-region[0]))
                if new_area > area and new_area > 0.1*box_area:
                    tar_face = face
                    tar_region = region
                    area = new_area
                else:
                    pass
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

            if len(same_unkwn_detected) > 15:
                print('New Person Detected')
                same_unkwn_detected.clear()
                id = len(self.face_db.index)
                name = str(id+111)+'jja'
                for emb in memory[-12:]:

                    entry = [id, name]
                    for i in emb:
                        entry.append(i)
                    print('Wriitng to DB',len(self.face_db.index))
                    with open(
                            '/home/nayan/PycharmProjects/CustomerRecognitionFunctionalModule/Face-Recognition-MIS/MIS_Face_DB.csv',
                            'a') as f_object:
                        writer_object = writer(f_object)
                        writer_object.writerow(entry)
                        f_object.close()
                print('successfully added the new person in the DB')
                print('Updating the application according to DB')
                self.face_db = pd.read_csv('MIS_Face_DB.csv')
                self.known_embeddings = self.face_db.iloc[:, 2:]
                known_labels = self.face_db.iloc[:, 0]
                self.known_labels = np.array(known_labels)
                self.names = self.face_db.iloc[:, 1]


            current_time = datetime.now().strftime("%H:%M:%S")
            # except:q
            #     print('Face Loading error, shape:', face.shape)

            detected_name = name
            print(same_unkwn_detected, 'name', detected_name)

            # print(face_region, face_region_resized, img_resized.shape)

            image_out = draw_boxes(org_img, tar_region, detected_name, scale)

            t2 = time.time()
            fps = round(1 / (t2 - t1), 1)
            cv2.rectangle(image_out, (40, 0), (300, 70), (0, 255, 0), -1)
            cv2.putText(image_out, f'FPS : {fps}', (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 128, 0),
                        thickness=2, lineType=1)
            return org_img
