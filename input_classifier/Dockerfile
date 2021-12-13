FROM python:3.7

WORKDIR app

COPY input_classifier/requirements.txt .

RUN pip install -r requirements.txt


COPY input_classifier/efficientnet-b0.pch input_classifier/
COPY input_classifier/*.py input_classifier/
COPY redis_storage.py .
COPY serialize.py .
COPY worker.py .

CMD python worker.py --handler=validator

