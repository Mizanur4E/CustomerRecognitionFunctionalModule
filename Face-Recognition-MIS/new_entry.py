from Retinaface_gpu.Retina_gpu import RetinaGPU
import cv2
import os
import glob


def capture_video(url, username):
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
    print('Video Capture Done...')


def video_to_images():
    video_dir = './all_videos/*'
    all_files = glob.glob(video_dir)
    destination_dir = './all_images'

    existed_names = glob.glob(destination_dir + '/*')
    existed_names = [i.split('/')[-1] for i in existed_names]

    process_files = []
    for nm in all_files:
        if not nm.split('/')[-1].split('.')[0] in existed_names:
            process_files.append(nm)
    print(f'Processing video for : {process_files}')
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
                for i in range(frame_rate // 2):
                    ret, frame = cap.read()
                if ret:
                    cv2.imwrite(destination_dir + f'/{name}/{count}.jpg', frame)
                    count += 1
                else:
                    break


def crop_images():
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

    process_names = []
    for nm in names:
        if not nm in existed:
            process_names.append(nm)

    print(f'Cropping faces for : {process_names}')
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
                saved_dir = output_dir + '/' + name + '/' + name + '_' + str(count) + '.jpg'
                face = cv2.resize(face, (112, 112))
                cv2.imwrite(saved_dir, face)
                print(file_path, saved_dir)

    print('All Faces Alignment Done...')


def main():
    url = 'rtsp://admin:AiBi@8899@192.168.101.80:554'
    username = input("Enter PersonName:")
    print("PersonName is: " + username)
    capture_video(url, username)
    video_to_images()
    crop_images()


if __name__ == '__main__':
    main()
