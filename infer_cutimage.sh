#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python3 inference_remote.py \
--trained_model weights/ssd512_MixShip512_10026000.pth \
--data_dir /data/03_Datasets/CasiaDatasets/MixShip512_100/ \
--annot_type rect \
--test_images /data/03_Datasets/CasiaDatasets/MixShip512_100/test.txt \
--save_folder CutShip/ssd512_MixShip18000_iter26000/test \
--conf_thre 0.3 \
--iou_thre 0.5 \
--re_evaluate False

