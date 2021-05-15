python3 eval_casia.py \
--annot_dir /data/03_Datasets/CasiaDatasets/CutShip/labelDota \
--annot_type rect \
--det_path CutShip/CutShip2_iter32000/test/detections.pkl \
--imagesetfile CutShip/CutShip2_iter32000/test/infer.imgnames \
--clss ship \
--iou_thre 0.3 \
--conf_thre 0.3
