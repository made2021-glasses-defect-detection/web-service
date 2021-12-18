FROM python:3.7

WORKDIR app

COPY validation/requirements.txt .

RUN pip install -r requirements.txt


COPY validation/efficientnet-b0.pch validation/
COPY validation/*.py validation/
COPY redis_storage.py .
COPY serialize.py .
COPY worker.py .

CMD python worker.py --handler=validator

