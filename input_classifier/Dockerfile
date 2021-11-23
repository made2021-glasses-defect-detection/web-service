FROM python:3.7

COPY requirements.txt .

RUN pip install -r requirements.txt

WORKDIR app
COPY efficientnet-b0.pch .
COPY classification_efficientnet.py .
COPY test_classification_efficientnet.py .
RUN pip install pytest