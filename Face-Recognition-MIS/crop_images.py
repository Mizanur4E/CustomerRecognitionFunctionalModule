from Retinaface_gpu.Retina_gpu import RetinaGPU
import cv2
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--all", default=False, action="store_true")
args = parser.parse_args()

dataset_dir = './all_images'
names = []
for name in os.listdir(dataset_dir):
    names.append(str(name))

output_dir = './cropped_dataset'
existed = []
for name in os.listdir(output_dir):
    existed.append(str(name))

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
else:
    print('path exists')


if args.all:
    process_names = names
else:
    process_names = []
    for nm in names:
        if not nm in existed:
            process_names.append(nm)

print(process_names)

retina = RetinaGPU(0)

for name in process_names:
    count = 0
    print('=======================================')
    sub_dir = dataset_dir + '/' + name
    saved_dir = output_dir + '/' + name
    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)
    print(saved_dir)

    for file in os.listdir(sub_dir):
        file_path = sub_dir + '/' + file
        image = cv2.imread(file_path)
        faces, region = retina.extract_faces(image, 0.3, alignment=True)

        for face in faces:
            count += 1
            saved_dir = output_dir+'/'+name+'/'+name+'_'+str(count)+'.jpg'
            face = cv2.resize(face, (112, 112))
            cv2.imwrite(saved_dir, face)
            print(file_path, saved_dir)

print('All Faces Alignment Done...')
