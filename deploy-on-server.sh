#!/bin/sh
git pull

docker-compose -f docker-compose.yml pull
docker-compose -f docker-compose.yml down --remove-orphans
docker-compose -f docker-compose.yml up -d

docker image prune -f
