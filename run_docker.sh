#!/bin/bash
SCRIPTPATH=$( cd "$(dirname "$0")" ; pwd -P )
# "
source $SCRIPTPATH/.env

docker run -d --name=predictor --restart=always -v $SCRIPTPATH:/tf tensorflow/tensorflow:latest-py3 \
    python /tf/one_image.py \
    --model=/tf/model_data/prod/ \
    --image=$CAMERA_URL \
    --repeat-interval=60 \
    --keep-history=10 \
    --save-history=/tf/history.json \
    --print
