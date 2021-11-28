# Web service

## Install dependencies

```sh
pip install -r requirements.txt
```

## Run for devlopment (with auto reloading)

```sh
pip install pyyaml
pip install watchdog -U
```

```sh
./dev-run.sh
```

## Build Image

```sh
docker build -f Dockerfile.web -t ghcr.io/made2021-glasses-defect-detection/web-server .
docker push ghcr.io/made2021-glasses-defect-detection/web-server
```

# Input classificator

## Test

```sh
python -m pytest test_classification_efficientnet.py
```

## Build Image

```sh
docker build -f Dockerfile.input_classifier -t ghcr.io/made2021-glasses-defect-detection/input-classifier .
docker push ghcr.io/made2021-glasses-defect-detection/input-classifier
```

## Test Image

```sh
docker run -it -v /E//Soft/MADE/semester_3/diploma/model/input_classifier:/app/images ghcr.io/made2021-glasses-defect-detection/input-classifier bash
```

# Docker compose 

## Local build and push to registry

```sh
docker-build-push.bat
```

## Local build and run


```sh
compose-run.bat
```