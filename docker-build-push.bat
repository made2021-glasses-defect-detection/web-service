#!/bin/sh

docker build -f Dockerfile -t ghcr.io/made2021-glasses-defect-detection/web-server .
docker build -f validation/Dockerfile -t ghcr.io/made2021-glasses-defect-detection/input-classifier .
docker build -f segmentation/Dockerfile -t ghcr.io/made2021-glasses-defect-detection/segmentation .
docker build -f classification/Dockerfile -t ghcr.io/made2021-glasses-defect-detection/classification .

docker push ghcr.io/made2021-glasses-defect-detection/web-server
docker push ghcr.io/made2021-glasses-defect-detection/input-classifier
docker push ghcr.io/made2021-glasses-defect-detection/segmentation
docker push ghcr.io/made2021-glasses-defect-detection/classification
