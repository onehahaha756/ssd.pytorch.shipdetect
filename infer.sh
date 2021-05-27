#!/bin/bash
python3 inference_remote.py \
--trained_model weights/ssd512_Airbus512_85000.pth \
--data_dir /data/03_Datasets/airbus-ship-detection/airbus_ship_detection512/ \
--annot_type rect \
--save_folder AirbusShip/ssd512_Airbus512_85000 \
--conf_thre 0.3 \
--iou_thre 0.5 \
--nms_thre 0.3 \
--re_evaluate True \
--test_images /data/03_Datasets/airbus-ship-detection/airbus_ship_detection512/test.txt \


