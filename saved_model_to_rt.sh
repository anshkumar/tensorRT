saved_model_cli convert \
--dir "/home/yilrr/tf-serving/faster-rcnn/saved_model/versions/1" \
--output_dir "/home/yilrr/tf-serving/trt-frcnn" \
--tag_set serve \
tensorrt --precision_mode FP32 --max_batch_size 32 --is_dynamic_op True
