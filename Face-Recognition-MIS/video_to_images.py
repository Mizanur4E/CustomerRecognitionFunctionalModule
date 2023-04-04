import cv2
import os
import glob
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--all", default=False, action="store_true")
args = parser.parse_args()

video_dir = './all_videos/*'
all_files = glob.glob(video_dir)
destination_dir = './all_images'

existed_names = glob.glob(destination_dir+'/*')
existed_names = [i.split('/')[-1] for i in existed_names]

if args.all:
    process_files = all_files
else:
    process_files = []
    for nm in all_files:
        if not nm.split('/')[-1].split('.')[0] in existed_names:
            process_files.append(nm)

print(process_files)

for file in process_files:
    name = file.split('/')[-1].split('.')[0]
    print(name)
    if not os.path.isdir(destination_dir + f'/{name}'):
        os.mkdir(destination_dir + f'/{name}')
    cap = cv2.VideoCapture(file)
    frame_rate = int(cap.get(5))
    count = 0
    if cap.isOpened():
        while True:
            for i in range(frame_rate//2):
                ret, frame = cap.read()
            if ret:
                cv2.imwrite(destination_dir + f'/{name}/{count}.jpg', frame)
                count += 1
            else:
                break

