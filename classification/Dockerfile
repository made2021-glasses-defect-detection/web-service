FROM python:3.7

WORKDIR app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY classification/requirements.txt .

RUN pip install -r requirements.txt

COPY classification/clf_whole.pth classification/
COPY classification/*.py classification/
COPY redis_storage.py .
COPY serialize.py .
COPY worker.py .

CMD python worker.py --handler=classification

