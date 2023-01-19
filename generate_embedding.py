import warnings
warnings.filterwarnings("ignore")
import os
import time
import pandas as pd
from Arcface_custom import ArcFaceRecog


print('Arcface with Resnet-50 as Backbone selected...')
arcface_model_path = './arc-models/model.tflite'
face_rec = ArcFaceRecog.ArcFace(arcface_model_path)
saved_embeddings = "embed_data_r50.csv"

dataset_dir = './cropped_dataset'
names = []
for name in os.listdir(dataset_dir):
    names.append(str(name))
print('names are : ', names)

t1 = time.time()
database = []
for name in names:
    print('=======================================')
    name_id = names.index(name)
    sub_dir = dataset_dir+'/'+name
    for file in os.listdir(sub_dir):
        file_path = sub_dir+'/'+file
        print(file_path)
        embd = list(face_rec.calc_emb(file_path))
        data = [int(name_id)] + [name] + embd
        database.append(data)

columns = ["ID", 'Name']
for i in range(512):
    columns.append(str(i))

Face_df = pd.DataFrame().from_records(database, columns=columns)
Face_df.to_csv('MIS_Face_DB.csv', index=False)

print('*********    Done Encoding    **********')
t2 = time.time()
print('Execution Time:  {} '.format(t2-t1))