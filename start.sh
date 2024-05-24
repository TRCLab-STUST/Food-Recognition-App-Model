#!/bin/bash
docker run -it --rm --user $(id -u):$(id -g) --shm-size 4G --gpus all -v .:/home/dd/yolov7/yolov7 -w /home/dd/yolov7/yolov7 tyson/yolov7:latest \
    python train.py \
    --workers 32 \
    --device 0 \
    --batch-size 32 \
    --data data/coco.yaml \
    --hyp data/hyp.scratch.custom.yaml \
    --img 640 640 \
    --cfg cfg/training/yolov7.yaml \
    --weights yolov7_training.pt \
    --name New26L__0519_lr00012_850e \
    --nosave \
    --epochs 850 \
    --cache-images \
    --image-weights \
    --sync-bn \


