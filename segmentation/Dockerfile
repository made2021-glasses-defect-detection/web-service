FROM python:3.7

WORKDIR app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY segmentation/requirements.txt .

RUN pip install -r requirements.txt

COPY segmentation/unet_res18_whole.pth segmentation/
COPY segmentation/*.py segmentation/
COPY redis_storage.py .
COPY serialize.py .
COPY worker.py .

CMD python worker.py --handler=segmentation

