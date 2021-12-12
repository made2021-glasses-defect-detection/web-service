#!/bin/sh

docker build -f Dockerfile.web -t ghcr.io/made2021-glasses-defect-detection/web-server .
docker build -f Dockerfile.input_classifier -t ghcr.io/made2021-glasses-defect-detection/input-classifier .
docker build -f Dockerfile.segmentation -t ghcr.io/made2021-glasses-defect-detection/segmentation .

docker push ghcr.io/made2021-glasses-defect-detection/web-server
docker push ghcr.io/made2021-glasses-defect-detection/input-classifier
docker push ghcr.io/made2021-glasses-defect-detection/segmentation
