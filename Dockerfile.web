FROM python:3.7

WORKDIR app

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN mkdir uploads
COPY static/ static/
COPY templates/ templates/
COPY redis_storage.py .
COPY server.py .

CMD gunicorn server:app

