#!/bin/bash

# Define the name and tag for your image
IMAGE_NAME="gcr.io/cloud-nas-260507/tpu_commons"
IMAGE_TAG=${USER}

# Build the Docker image
# On the first time, you may need 'gcloud auth configure-docker'
docker build -f docker/Dockerfile -t "${IMAGE_NAME}:${IMAGE_TAG}" .


docker push "${IMAGE_NAME}:${IMAGE_TAG}"
