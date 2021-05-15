#!/bin/bash
python3 inference_remote.py \
--trained_model weights/ssd512_MixShip512_10058000.pth \
--data_dir /data/03_Datasets/CasiaDatasets/ship/ \
--annot_type polygon \
--save_folder Ship/ssd512_MixShip1800_iter58000 \
--conf_thre 0.3 \
--iou_thre 0.5 \
--re_evaluate False

