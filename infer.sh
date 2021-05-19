#!/bin/bash
python3 inference_remote.py \
--trained_model weights/ssd512_MixCutShip512_300_46000.pth \
--data_dir /data/03_Datasets/CasiaDatasets/ship/ \
--annot_type polygon \
--save_folder Ship/ssd512_MixShip300_iter46000_test2 \
--test_images /data/03_Datasets/CasiaDatasets/CutMixShip512_300/origin_test.txt \
--conf_thre 0.3 \
--iou_thre 0.5 \
--nms_thre 0.3 \
--re_evaluate False

