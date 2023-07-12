import cv2
import numpy as np
import time
from Retinaface_gpu.face_detection import RetinaFace
from Retinaface_gpu.align import alignment_procedure

class RetinaGPU():
    def __init__(self, gpuID):
        
        self.detector = RetinaFace(gpuID)
        print('Retina GPU Model with Resnet-50 as Backbone Loaded...')

    def map_value(self, area, limit):
        if len(area)==len(limit):
            b=[]
            for j in range(len(area)):
                x=area[j]
                lim=limit[j]
                x=max(0,x)
                x=min(x,lim)
                b.append(x)
        b=np.array(b)
        return b


    def extract_faces(self, image, scale, alignment=True):

        img = image.copy()
        resized_image = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        faces = self.detector(resized_image)

        regions=[]
        cropped_faces=[]
        for i in range (len(faces)):
            
            limits=[img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
            box, landmarks, score = faces[i]
            #landmarks = landmarks.astype(np.int)
            landmarks =  np.round(landmarks/scale).astype(int)
            
            box = box.astype(np.int)
            regions.append([box[0], box[1], box[2], box[3]])
            facial_area=np.array([box[0], box[1], box[2], box[3]])
           
            facial_area = np.round(np.array(facial_area)/scale).astype(int)
            facial_area = self.map_value(facial_area,limits)
           
            facial_img = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]

            left_eye = landmarks[1]
            right_eye = landmarks[0]
            nose = landmarks[2]
            
            if alignment==True:
                facial_img = alignment_procedure(facial_img, right_eye, left_eye, nose)
            
            cropped_faces.append(facial_img)

        return cropped_faces, regions


