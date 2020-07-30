#!/usr/bin/env bash

# Build image
docker build --tag=mnistserver .

# List docker images
docker image ls

# Run flask app
docker run -p 8848:8848 mnistserver