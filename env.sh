#!/bin/bash
docker run -it --rm --user $(id -u):$(id -g) --shm-size 2G --gpus all -v .:/home/dd/yolov7/yolov7 -w /home/dd/yolov7/yolov7 tyson/yolov7:latest
