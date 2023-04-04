# Face-Recognition-MIS

## Face Detection
- [x] RetinFace

## Face Recognition
- [x] ArcFace (Resnet-50 Tflite)

## New Person Entry
Run 

`python new_entry.py` to capture video

`python generate_embeddings.py` to generate embeddings from cropped images

## Run Server

```
ssh aci-mis-ai@192.168.101.63

cd Face-Recognition-MIS/

source venv/bin/activate

Python main.py –-server
```


