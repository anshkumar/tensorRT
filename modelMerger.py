##################################
# author: vedanshu
##################################


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# os.environ["TF_ENABLE_WHILE_V2"]="1" 
# os.environ["TF_ENABLE_COND_V2"]="1"
import tensorflow as tf
import numpy as np
import cv2
import os
import shutil

grid_shape = (5,5) # (width, height)
image_shape=(150, 150) # (width, height) Width has to be equal to height
num_channels=3
MAX_DETECTION_IN_L2 = 10
L1_SCORE_THRESH = 0.3
L2_SCORE_THRESH = 0.3
L1_MASK_THRESH = 0.3
IS_L1_MASK = False
IS_L2_MASK = False
output_saved_model_dir = "saved_model"
pb_fname1 = "/Users/vedanshu/frozen_graph/ved_potato_l1_sort_ssdlite_mobilenet_edgetpu.pb"
pb_fname2 = "/Users/vedanshu/frozen_graph/ved_potato_l2_5x5_sort_faster_rcnn_inception_v2.pb"

use_trt =False

if use_trt:
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    precision_mode='INT8' # FP16 or FP32 or INT8
    output_saved_model_dir_rt = "/home/xavier3/ved_potato_single_ssd_faster_rt"
    tf_calib_data_files = "/home/xavier3/ved_test.record"

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.GFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

def rename_frame_name(graphdef, suffix):
    # Bug reported at https://github.com/tensorflow/tensorflow/issues/22162#issuecomment-428091121
    for n in graphdef.node:
        if "while" in n.name:
            if "frame_name" in n.attr:
                n.attr["frame_name"].s = str(n.attr["frame_name"]).replace("while_context",
                                                                           "while_context" + suffix).encode('utf-8')

l1_graph = tf.Graph()
with l1_graph.as_default():
    trt_graph1 = get_frozen_graph(pb_fname1)
    if IS_L1_MASK:
        [tf_input1, tf_scores1, tf_boxes1, tf_classes1, tf_num_detections1, tf_masks1] = tf.import_graph_def(trt_graph1, 
                return_elements=['image_tensor:0', 'detection_scores:0', 'detection_boxes:0', 'detection_classes:0','num_detections:0', 'detection_masks:0'])
    else:
        [tf_input1, tf_scores1, tf_boxes1, tf_classes1, tf_num_detections1] = tf.import_graph_def(trt_graph1, 
                return_elements=['image_tensor:0', 'detection_scores:0', 'detection_boxes:0', 'detection_classes:0','num_detections:0'])
        
    input1 = tf.identity(tf_input1, name="l1_input")
    boxes1 = tf.identity(tf_boxes1[0], name="l1_boxes")  # index by 0 to remove batch dimension
    scores1 = tf.identity(tf_scores1[0], name="l1_scores")
    classes1 = tf.identity(tf_classes1[0], name="l1_classes")
    num_detections1 = tf.identity(tf.dtypes.cast(tf_num_detections1[0], tf.int32), name="l1_num_detections")
    if IS_L1_MASK:
        masks1 = tf.identity(tf_masks1[0], name="l1_masks")
    
intermediate_graph = tf.Graph()
with intermediate_graph.as_default():
    def image_grid(input_tensor, grid_shape=(5,5), image_shape=(150, 150), num_channels=3):
        # https://github.com/tensorflow/tensorflow/blob/23c218785eac5bfe737eec4f8081fd0ef8e0684d/tensorflow/contrib/gan/python/eval/python/eval_utils_impl.py#L34
        height, width = grid_shape[0] * image_shape[0], grid_shape[1] * image_shape[1]
        input_tensor = tf.reshape(
          input_tensor, tuple(grid_shape) + tuple(image_shape) + (num_channels,))
        input_tensor = tf.transpose(input_tensor, [0, 1, 3, 2, 4])
        input_tensor = tf.reshape(
          input_tensor, [grid_shape[0], width, image_shape[0], num_channels])
        input_tensor = tf.transpose(input_tensor, [0, 2, 1, 3])
        input_tensor = tf.reshape(
          input_tensor, [height, width, num_channels])
        return input_tensor

    def get_grid_roies():
        def condition1(i, k, boxes_pixels, inds_boxes_pixels):
            return tf.less(i, num_detections1)
        def body1(i, k, boxes_pixels, inds_boxes_pixels):
            def get_boxes_pixels(k, boxes_pixels, inds_boxes_pixels):
                normalizer = [tf.shape(tf_input1[0])[0], tf.shape(tf_input1[0])[1], tf.shape(tf_input1[0])[0], tf.shape(tf_input1[0])[1]]
                box = tf.multiply(boxes1[i], tf.dtypes.cast(normalizer, tf.float32))
                box = tf.dtypes.cast(tf.round(box), tf.int32)
                boxes_pixels = boxes_pixels.write(k, box)
                inds_boxes_pixels = inds_boxes_pixels.write(k, i)
                return tf.add(k, 1), boxes_pixels, inds_boxes_pixels
                
            def return_null_boxes_pixels(k, boxes_pixels, inds_boxes_pixels):
                return k, boxes_pixels, inds_boxes_pixels
            
            k, boxes_pixels, inds_boxes_pixels = tf.cond(tf.greater_equal(scores1[i], L1_SCORE_THRESH), 
                                                         lambda: get_boxes_pixels(k, boxes_pixels, inds_boxes_pixels), 
                                                         lambda: return_null_boxes_pixels(k, boxes_pixels, inds_boxes_pixels))
            return [tf.add(i, 1), k, boxes_pixels, inds_boxes_pixels]

        i = tf.constant(0)
        k = tf.constant(0)
        boxes_pixels = tf.TensorArray(dtype=tf.int32,size=1,dynamic_size=True,clear_after_read=False, name='boxes_pixels')
        inds_boxes_pixels = tf.TensorArray(dtype=tf.int32,size=1,dynamic_size=True,clear_after_read=False, name='inds_boxes_pixels')

        _,_, boxes_pixels, inds_boxes_pixels = tf.while_loop(condition1,body1,[i, k, boxes_pixels, inds_boxes_pixels])
        boxes_pixels = boxes_pixels.stack()
        inds_boxes_pixels = inds_boxes_pixels.stack()

        def condition2(j, boxes_pixels, roies):
            return tf.less(j, tf.shape(boxes_pixels)[0])
        def body2(j, boxes_pixels, roies):
            startY =  boxes_pixels[j][0]
            startX =  boxes_pixels[j][1]
            endY =  boxes_pixels[j][2]
            endX =  boxes_pixels[j][3]
            boxW = endX - startX
            boxH = endY - startY
            
            if IS_L1_MASK:
                mask = masks1[j]
                mask = (mask > L1_MASK_THRESH)
                mask = tf.stack([mask, mask, mask],axis=2)
                mask = tf.cast(mask, tf.uint8)
                mask = tf.image.resize(mask, (boxH, boxW))
                roi = tf.multiply(tf_input1[0, startY:endY, startX:endX], tf.cast(mask, tf.uint8))
            else:
                roi = tf_input1[0, startY:endY, startX:endX] # batch: 0
            roi = tf.image.resize_image_with_pad(roi,image_shape[0],image_shape[1])
            roi = tf.dtypes.cast(roi, tf.uint8)
            roies = roies.write(j, roi)
            return [tf.add(j, 1), boxes_pixels, roies]

        j = tf.constant(0)
        roies = tf.TensorArray(dtype=tf.uint8,size=1,dynamic_size=True,clear_after_read=False,
                               infer_shape=False, name='roies')

        _, _, roies = tf.while_loop(condition2,body2,[j, boxes_pixels, roies])

        # Adding padding for making grid
        roies = roies.stack()
        zero_pad = tf.zeros([1,image_shape[0],image_shape[1],num_channels], tf.uint8)
        _no_pad = tf.mod(tf.shape(roies)[0], tf.constant(grid_shape[0]*grid_shape[1]))
        no_pad = tf.cond(tf.equal(_no_pad, tf.constant(0)), lambda: tf.constant(0), 
                         lambda: tf.subtract(tf.constant(grid_shape[0]*grid_shape[1]), _no_pad))
        zero_pad = tf.tile(zero_pad, [no_pad,1,1,1])
        roies = tf.concat([roies, zero_pad], axis=0)

        # Creating batch of images of size grid_shape[0]*image_shape[0]
        size = grid_shape[0]*grid_shape[1]
        n_iter = tf.dtypes.cast(tf.shape(roies)[0]/size, tf.float32) #  int32 / int32 = float64 which leads 49/49 = 0.99999999999999989
        n_iter = tf.dtypes.cast(n_iter, tf.int32)
        
        k = tf.constant(0)
        grid_roies = tf.TensorArray(dtype=tf.uint8,size=1,dynamic_size=True,clear_after_read=False,infer_shape=False, name='grid_roies')

        def condition3(k, grid_roies):
            return tf.less(k, n_iter)

        def body3(k, grid_roies):
            grid_roi = image_grid(roies[size*k:size*(k+1)], grid_shape, image_shape, num_channels)
            grid_roies = grid_roies.write(k, grid_roi)
            return [tf.add(k, 1), grid_roies]

        _, grid_roies = tf.while_loop(condition3, body3, [k, grid_roies])

        grid_roies = grid_roies.stack() 

        return grid_roies, n_iter, inds_boxes_pixels

    def create_empty_grid():
        grid_roies = tf.zeros([1,image_shape[0]*grid_shape[0],image_shape[1]*grid_shape[1],num_channels], tf.uint8)
        return grid_roies, 1, tf.range(grid_shape[0]*grid_shape[1])

    l1_graph_def = l1_graph.as_graph_def()
    g1name = "level1"
    rename_frame_name(l1_graph_def, g1name)
    tf.import_graph_def(l1_graph_def, name=g1name)

    tf_input1 = tf.get_default_graph().get_tensor_by_name('level1/l1_input:0')
    boxes1 = tf.get_default_graph().get_tensor_by_name('level1/l1_boxes:0')
    scores1 = tf.get_default_graph().get_tensor_by_name('level1/l1_scores:0')
    classes1 = tf.get_default_graph().get_tensor_by_name('level1/l1_classes:0')
    num_detections1 = tf.get_default_graph().get_tensor_by_name('level1/l1_num_detections:0')
    if IS_L1_MASK:
        masks1 = tf.get_default_graph().get_tensor_by_name('level1/l1_masks:0')
    
    _detections1 = tf.count_nonzero(tf.greater_equal(scores1, L1_SCORE_THRESH), dtype=tf.int32)

    # Grid is filled along the column first
    grid_roies, batch_size, inds_boxes_pixels = tf.cond(tf.equal(_detections1, tf.constant(0)), create_empty_grid, get_grid_roies)
    grid_roies = tf.identity(grid_roies, name="grid_roies_out")
    batch_size = tf.identity(batch_size, name="batch_size")
    
    boxes1 = tf.gather(boxes1, inds_boxes_pixels, name="l1_boxes")
    scores1 = tf.gather(scores1, inds_boxes_pixels, name="l1_scores")
    classes1 = tf.gather(classes1, inds_boxes_pixels, name="l1_classes")
    if IS_L1_MASK:
        masks1 = tf.gather(masks1, inds_boxes_pixels, name="l1_masks")
    
    tf_input1 = tf.get_default_graph().get_tensor_by_name("level1/import/image_tensor:0")

connected_graph = tf.Graph()
tf_sess_main = tf.Session(graph=connected_graph)

with connected_graph.as_default():
    intermediate_graph_def = intermediate_graph.as_graph_def()
    g1name = 'ved'
    rename_frame_name(intermediate_graph_def, g1name)
    tf.import_graph_def(intermediate_graph_def, name=g1name)
    
    tf_input = tf.get_default_graph().get_tensor_by_name('ved/level1/import/image_tensor:0')
    tf_grid_out = tf.get_default_graph().get_tensor_by_name('ved/grid_roies_out:0')
    tf_batch_size = tf.get_default_graph().get_tensor_by_name('ved/batch_size:0')
    tf_num_detections_l1 = tf.dtypes.cast(tf.get_default_graph().get_tensor_by_name('ved/level1/import/num_detections:0'), tf.int32)
    tf_boxes_l1 = tf.get_default_graph().get_tensor_by_name('ved/l1_boxes:0')
    tf_scores_l1 = tf.get_default_graph().get_tensor_by_name('ved/l1_scores:0')
    tf_classes_l1 = tf.get_default_graph().get_tensor_by_name('ved/l1_classes:0')
    if IS_L1_MASK:
        tf_masks_l1 = tf.get_default_graph().get_tensor_by_name('ved/l1_masks:0')
    
    trt_graph2 = get_frozen_graph(pb_fname2)
    g2name = 'level2'
    rename_frame_name(trt_graph2, g2name)
    [tf_scores, tf_boxes, tf_classes, tf_num_detections] = tf.import_graph_def(trt_graph2,
            input_map={'image_tensor': tf_grid_out},
            return_elements=['detection_scores:0', 'detection_boxes:0', 'detection_classes:0','num_detections:0'])    
        
    ip_image_height = tf.shape(tf_input[0])[0]
    ip_image_width = tf.shape(tf_input[0])[1]
    boxes_l1 = tf_boxes_l1 * tf.dtypes.cast([ip_image_height, ip_image_width, ip_image_height, ip_image_width], tf.float32)
    boxes_l1 = tf.dtypes.cast(boxes_l1, tf.int32)
    tf_boxes_l1 = tf.identity(boxes_l1)
#     tf_boxes_l1 = tf.expand_dims(boxes_l1, 0)
    
    def getSquareImageFactors( img_width, img_height, target_width ):
        max_dim = tf.cond(img_width >= img_height, lambda: img_width, lambda: img_height)
        scale = target_width / max_dim
        
        def getScaledHeight(img_width, img_height, target_width, scale):
            x = 0
            img_height = tf.dtypes.cast(tf.dtypes.cast(img_height, tf.float64) * scale, tf.int32)
            y = tf.dtypes.cast(( target_width - img_height ) / 2, tf.int32)
            return x,y
        def getScaledWidth(img_width, img_height, target_width, scale):
            y = 0
            img_width = tf.dtypes.cast(tf.dtypes.cast(img_width, tf.float64) * scale, tf.int32)
            x = tf.dtypes.cast(( target_width - img_width ) / 2, tf.int32)
            return x,y
        
        x, y = tf.cond(img_width >= img_height, lambda: getScaledHeight(img_width, img_height, target_width, scale), 
                       lambda: getScaledWidth(img_width, img_height, target_width, scale))
        
        return x, y, scale
        
    def condition1(i, tf_boxes_l2, tf_scores_l2, tf_classes_l2, batch_arr_original_index):
        return tf.less(i, tf_batch_size)
    def body1(i, tf_boxes_l2, tf_scores_l2, tf_classes_l2, batch_arr_original_index):
        boxes_l2 = tf_boxes[i]
        scores_l2 = tf_scores[i]
        classes_l2 = tf_classes[i]
        num_detections_l2 = tf.dtypes.cast(tf_num_detections[i], tf.int32)
        
        grid_height = image_shape[0]*grid_shape[0]
        grid_width = image_shape[1]*grid_shape[1]
        boxes_l2 = boxes_l2 * tf.dtypes.cast(tf.constant([grid_height, grid_width, grid_height, grid_width]), tf.float32)
        boxes_index = boxes_l2/tf.dtypes.cast(tf.constant(image_shape[0]), tf.float32)
        boxes_index = tf.dtypes.cast(boxes_index, tf.int32)
        boxes_l2 = tf.dtypes.cast(boxes_l2, tf.int32)
        
        
        def condition2(j, arr_boxes_l2, arr_original_index, arr_grid_original_index):
            return tf.less(j, tf.shape(boxes_index)[0])
        def body2(j, arr_boxes_l2, arr_original_index, arr_grid_original_index):
            def update_boxes(arr_boxes_l2, arr_original_index, arr_grid_original_index):
                y = boxes_index[j][0]
                x = boxes_index[j][1]
                original_index = i*grid_shape[0]*grid_shape[1] + y*tf.constant(grid_shape[0]) + x # indexed zero
                grid_original_index = y*tf.constant(grid_shape[0]) + x
                
                arr_original_index = arr_original_index.write(j, original_index)
                arr_grid_original_index = arr_grid_original_index.write(j, grid_original_index)
                
                _x, _y, _scale = getSquareImageFactors(boxes_l1[original_index][3] - boxes_l1[original_index][1],
                                               boxes_l1[original_index][2] - boxes_l1[original_index][0],
                                               image_shape[1] )
                
                _y1 = tf.dtypes.cast(tf.dtypes.cast(boxes_l2[j][0] - y*tf.constant(image_shape[1]) - _y, tf.float64) / _scale, tf.int32)
                _x1 = tf.dtypes.cast(tf.dtypes.cast(boxes_l2[j][1] - x*tf.constant(image_shape[0]) - _x, tf.float64) / _scale, tf.int32)
                _y2 = tf.dtypes.cast(tf.dtypes.cast(boxes_l2[j][2] - y*tf.constant(image_shape[1]) - _y, tf.float64) / _scale, tf.int32)
                _x2 = tf.dtypes.cast(tf.dtypes.cast(boxes_l2[j][3] - x*tf.constant(image_shape[0]) - _x, tf.float64) / _scale, tf.int32)

                arr_boxes_l2 = arr_boxes_l2.write(j, [boxes_l1[original_index][0] + _y1,
                                                      boxes_l1[original_index][1] + _x1,
                                                      boxes_l1[original_index][0] + _y2,
                                                      boxes_l1[original_index][1] + _x2])

                return [arr_boxes_l2, arr_original_index, arr_grid_original_index]

            def update_with_null(arr_boxes_l2, arr_original_index, arr_grid_original_index):
                y = boxes_index[j][0]
                x = boxes_index[j][1]
                original_index = i*grid_shape[0]*grid_shape[1] + y*tf.constant(grid_shape[0]) + x
                grid_original_index = y*tf.constant(grid_shape[0]) + x
                arr_original_index = arr_original_index.write(j, original_index)
                arr_grid_original_index = arr_grid_original_index.write(j, grid_original_index)
                arr_boxes_l2 = arr_boxes_l2.write(j,[0,0,0,0])
                return [arr_boxes_l2, arr_original_index, arr_grid_original_index]

            arr_boxes_l2, arr_original_index, arr_grid_original_index = tf.cond(
                tf.math.logical_and(tf.greater_equal(scores_l2[j], L2_SCORE_THRESH),
                                    tf.math.logical_and(tf.math.equal(boxes_index[j][0],boxes_index[j][2]),
                                                        tf.math.equal(boxes_index[j][1],boxes_index[j][3]))),
                lambda: update_boxes(arr_boxes_l2, arr_original_index, arr_grid_original_index), 
                lambda: update_with_null(arr_boxes_l2, arr_original_index, arr_grid_original_index),
                name='if_cond_update')
            return [tf.add(j, 1), arr_boxes_l2, arr_original_index, arr_grid_original_index]

        j = tf.constant(0)
        arr_boxes_l2 = tf.TensorArray(dtype=tf.int32,size=1, dynamic_size=True,clear_after_read=False)
        arr_original_index = tf.TensorArray(dtype=tf.int32,size=1, dynamic_size=True,clear_after_read=False)
        arr_grid_original_index = tf.TensorArray(dtype=tf.int32,size=1, dynamic_size=True,clear_after_read=False)
        j, arr_boxes_l2, arr_original_index, arr_grid_original_index = tf.while_loop(condition2, 
                                                                                     body2, 
                                                                                     [j, arr_boxes_l2, arr_original_index, arr_grid_original_index],
                                                                                    name='while_over_l2_boxes_index')
        boxes_l2 = arr_boxes_l2.stack()
        original_index = arr_original_index.stack()
        grid_original_index = arr_grid_original_index.stack()
        
        inds = tf.argsort(grid_original_index,axis=-1,direction='ASCENDING',stable=False,name=None)
        boxes_l2 = tf.gather(boxes_l2, inds)
        grid_original_index = tf.gather(grid_original_index, inds)
        original_index = tf.gather(original_index, inds)
        scores_l2 = tf.gather(scores_l2, inds)
        classes_l2 = tf.gather(classes_l2, inds)
        
        grid_original_index, idx = tf.unique(grid_original_index)
        original_index, _ = tf.unique(original_index)
        
        partitioned_boxes_l2 = tf.dynamic_partition(boxes_l2, idx, grid_shape[0]*grid_shape[1])
        partitioned_scores_l2 = tf.dynamic_partition(scores_l2, idx, grid_shape[0]*grid_shape[1])
        partitioned_classes_l2 = tf.dynamic_partition(classes_l2, idx, grid_shape[0]*grid_shape[1])
        
        _x = tf.constant(grid_shape[0]*grid_shape[1]) - tf.shape(original_index)[0]
        _r = tf.cond(tf.less(_x, 0), lambda: tf.constant(0), lambda: tf.identity(_x))
        pad = tf.tile([tf_batch_size*grid_shape[0]*grid_shape[1]], [_r]) # the last index (tf_batch_size*grid_shape[0]*grid_shape[1]) will be excluded during scatter_nd
        original_index = tf.concat([original_index,pad], 0)

        for u in range(len(partitioned_boxes_l2)):
            _x = tf.constant(MAX_DETECTION_IN_L2) - tf.shape(partitioned_boxes_l2[u])[0]
            _r = tf.cond(tf.less(_x, 0), lambda: tf.constant(0), lambda: tf.identity(_x))
            pad = tf.tile(tf.zeros([1,4], tf.int32), [_r, 1])
            partitioned_boxes_l2[u] = tf.concat([partitioned_boxes_l2[u][:MAX_DETECTION_IN_L2],pad], 0)

        for u in range(len(partitioned_scores_l2)):
            _x = tf.constant(MAX_DETECTION_IN_L2) - tf.shape(partitioned_scores_l2[u])[0]
            _r = tf.cond(tf.less(_x, 0), lambda: tf.constant(0), lambda: tf.identity(_x))
            pad = tf.tile(tf.zeros([1], tf.float32), [_r])
            partitioned_scores_l2[u] = tf.concat([partitioned_scores_l2[u][:MAX_DETECTION_IN_L2],pad], 0)
            
        for u in range(len(partitioned_classes_l2)):
            _x = tf.constant(MAX_DETECTION_IN_L2) - tf.shape(partitioned_classes_l2[u])[0]
            _r = tf.cond(tf.less(_x, 0), lambda: tf.constant(0), lambda: tf.identity(_x))
            pad = tf.tile(tf.zeros([1], tf.float32), [_r])
            partitioned_classes_l2[u] = tf.concat([partitioned_classes_l2[u][:MAX_DETECTION_IN_L2],pad], 0)

        partitioned_boxes_l2 = tf.convert_to_tensor(partitioned_boxes_l2)
        partitioned_scores_l2 = tf.convert_to_tensor(partitioned_scores_l2)
        partitioned_classes_l2 = tf.convert_to_tensor(partitioned_classes_l2)
        
        batch_arr_original_index = batch_arr_original_index.write(i, original_index)
        tf_boxes_l2 = tf_boxes_l2.write(i, partitioned_boxes_l2)
        tf_scores_l2 = tf_scores_l2.write(i, partitioned_scores_l2)
        tf_classes_l2 = tf_classes_l2.write(i, partitioned_classes_l2)
        
        return [tf.add(i, 1), tf_boxes_l2, tf_scores_l2, tf_classes_l2, batch_arr_original_index]
        
    i = tf.constant(0)
    tf_boxes_l2 = tf.TensorArray(dtype=tf.int32,size=1,dynamic_size=True,clear_after_read=False, name='tf_arr_boxes_l2')
    tf_scores_l2 = tf.TensorArray(dtype=tf.float32,size=1,dynamic_size=True,clear_after_read=False, name='tf_arr_scores_l2')
    tf_classes_l2 = tf.TensorArray(dtype=tf.float32,size=1,dynamic_size=True,clear_after_read=False, name='tf_arr_classes_l2')
    batch_arr_original_index = tf.TensorArray(dtype=tf.int32,size=1, dynamic_size=True,clear_after_read=False)
    _, tf_boxes_l2, tf_scores_l2, tf_classes_l2, batch_arr_original_index = tf.while_loop(condition1, 
                                                                                          body1, 
                                                                                          [i, tf_boxes_l2, tf_scores_l2, tf_classes_l2, batch_arr_original_index],
                                                                                         name='while_over_batch_size')
    
    tf_boxes_l2 = tf_boxes_l2.stack()
    tf_scores_l2 = tf_scores_l2.stack()
    tf_classes_l2 = tf_classes_l2.stack()
    batch_arr_original_index = batch_arr_original_index.stack()
       
    tf_original_index_l2 = tf.squeeze(batch_arr_original_index, name="original_index_l2")
    
    tf_boxes_l2 = tf.reshape(tf_boxes_l2, [-1,MAX_DETECTION_IN_L2,4])
    tf_scores_l2 = tf.reshape(tf_scores_l2, [-1, MAX_DETECTION_IN_L2])
    tf_classes_l2 = tf.reshape(tf_classes_l2, [-1, MAX_DETECTION_IN_L2])
    
    tf_boxes_l1 = tf.identity(tf_boxes_l1, name="detection_boxes_l1")
    tf_scores_l1 = tf.identity(tf_scores_l1, name="detection_scores_l1")
    tf_classes_l1 = tf.identity(tf_classes_l1, name="detection_classes_l1")
    if IS_L1_MASK:
        tf_masks_l1 = tf.identity(tf_masks_l1, name="detection_masks_l1")
    
    _indices = tf.reshape(tf_original_index_l2, [-1]) 
    _indices = tf.expand_dims(_indices, 1)
    
    _shape_boxes_l2 = [tf_batch_size*grid_shape[0]*grid_shape[1] + 1, MAX_DETECTION_IN_L2, 4]
    tf_boxes_l2 = tf.scatter_nd(_indices, tf_boxes_l2, _shape_boxes_l2, name="detection_boxes_l2_scatter_nd")
    tf_boxes_l2 = tf_boxes_l2[:tf.shape(tf_boxes_l1)[0]]
    tf_boxes_l2 = tf.identity(tf_boxes_l2, name="detection_boxes_l2")
    
    _shape_scores_l2 = [tf_batch_size*grid_shape[0]*grid_shape[1] + 1, MAX_DETECTION_IN_L2]     
    tf_scores_l2 = tf.scatter_nd(_indices, tf_scores_l2, _shape_scores_l2, name="detection_scores_l2_scatter_nd")
    tf_scores_l2 = tf_scores_l2[:tf.shape(tf_boxes_l1)[0]]
    tf_scores_l2 = tf.identity(tf_scores_l2, name="detection_scores_l2")
    
    _shape_classes_l2 = [tf_batch_size*grid_shape[0]*grid_shape[1] + 1, MAX_DETECTION_IN_L2]
    tf_classes_l2 = tf.scatter_nd(_indices, tf_classes_l2, _shape_classes_l2, name="detection_classes_l2_scatter_nd")
    tf_classes_l2 = tf_classes_l2[:tf.shape(tf_boxes_l1)[0]]
    tf_classes_l2 = tf.identity(tf_classes_l2, name="detection_classes_l2")
    
    tf_max_num_detection = tf.identity(tf.shape(tf_boxes_l1)[0], name="max_num_detections")
                    
                                    
with connected_graph.as_default():
    print('\nSaving...')
    # cwd = os.getcwd()
    # path = os.path.join(cwd, 'saved_model')
    path = output_saved_model_dir
    shutil.rmtree(path, ignore_errors=True)
    inputs_dict = {
        "image_tensor": tf_input
    }
    if IS_L1_MASK:
        outputs_dict = {
            "detection_boxes_l1": tf_boxes_l1,
            "detection_scores_l1": tf_scores_l1,
            "detection_classes_l1": tf_classes_l1,
            "detection_masks_l1": tf_masks_l1,
            "max_num_detection": tf_max_num_detection,
            "detection_boxes_l2": tf_boxes_l2,
            "detection_scores_l2": tf_scores_l2,
            "detection_classes_l2": tf_classes_l2
        }
    else:
        outputs_dict = {
            "detection_boxes_l1": tf_boxes_l1,
            "detection_scores_l1": tf_scores_l1,
            "detection_classes_l1": tf_classes_l1,
            "max_num_detection": tf_max_num_detection,
            "detection_boxes_l2": tf_boxes_l2,
            "detection_scores_l2": tf_scores_l2,
            "detection_classes_l2": tf_classes_l2
        }
    tf.saved_model.simple_save(
        tf_sess_main, path, inputs_dict, outputs_dict
    )
    print('Ok')

def my_next_data():
    for i in range(10):
        yield tf.random.normal([1, 1200, 1200, 3])

def read_tfrecord(serialized_example):
    # https://github.com/tensorflow/models/blob/bcb231f00c252f4525d4a60be1dd6c605a296b6a/official/vision/detection/dataloader/tf_example_decoder.py
    feature_description = {
        'image/encoded':
            tf.io.FixedLenFeature((), tf.string),
        'image/source_id':
            tf.io.FixedLenFeature((), tf.string),
        'image/height':
            tf.io.FixedLenFeature((), tf.int64),
        'image/width':
            tf.io.FixedLenFeature((), tf.int64),
        'image/object/bbox/xmin':
            tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax':
            tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin':
            tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax':
            tf.io.VarLenFeature(tf.float32),
        'image/object/class/label':
            tf.io.VarLenFeature(tf.int64),
        'image/object/area':
            tf.io.VarLenFeature(tf.float32),
        'image/object/is_crowd':
            tf.io.VarLenFeature(tf.int64),
    }
    parsed_tensors = tf.io.parse_single_example(serialized_example, feature_description)
    image = tf.io.decode_image(parsed_tensors['image/encoded'], channels=3)
    image.set_shape([None, None, 3]) 
    return image

# dataset = tf.data.TFRecordDataset(tf_calib_data_files)
# parsed_dataset = dataset.map(read_tfrecord)
# iterator = parsed_dataset.make_one_shot_iterator()
# features = iterator.get_next()

def input_map():
    return {'ved/level1/import/image_tensor:0': features}

'''
"TrtConversionParams",
    [

        # A template RewriterConfig proto used to create a TRT-enabled
        # RewriterConfig. If None, it will use a default one.
        "rewriter_config_template",

        # The maximum GPU temporary memory which the TRT engine can use at
        # execution time. This corresponds to the 'workspaceSize' parameter of
        # nvinfer1::IBuilder::setMaxWorkspaceSize().
        "max_workspace_size_bytes",

        # One of TrtPrecisionMode.supported_precision_modes().
        "precision_mode",

        # The minimum number of nodes required for a subgraph to be replaced by
        # TRTEngineOp.
        "minimum_segment_size",

        # Whether to generate dynamic TRT ops which will build the TRT network
        # and engine at run time.
        #
        # TODO(laigd): In TF 2.0, this options should only affect INT8 mode.
        "is_dynamic_op",

        # Max number of cached TRT engines in dynamic TRT ops. If the number of
        # cached engines is already at max but none of them can serve the input,
        # the TRTEngineOp will fall back to run the TF function based on which
        # the TRTEngineOp is created.
        "maximum_cached_engines",

        # This argument is ignored if precision_mode is not INT8. If set to
        # True, a calibration graph will be created to calibrate the missing
        # ranges. The calibration graph must be converted to an inference graph
        # by running calibration with calibrate(). If set to False, quantization
        # nodes will be expected for every tensor in the graph (exlcuding those
        # which will be fused). If a range is missing, an error will occur.
        # Please note that accuracy may be negatively affected if there is a
        # mismatch between which tensors TRT quantizes and which tensors were
        # trained with fake quantization.
        "use_calibration",

        # If set to True, it will create a FunctionDef for each subgraph that is
        # converted to TRT op, and if TRT ops fail to execute at runtime, it'll
        # invoke that function as a fallback.
        "use_function_backup",

        # Max size for the input batch.
        # This option is deprecated in TF 2.0.
        "max_batch_size",

        # A list of batch sizes used to create cached engines, only used when
        # is_dynamic_op is True. The length of the list should be <=
        # maximum_cached_engines, and the dynamic TRT op will use this list to
        # determine the batch sizes of the cached engines, instead of making the
        # decision on the fly. This is useful when we know the most common batch
        # size(s) the application is going to generate.
        # This option is deprecated in TF 2.0.
        "cached_engine_batches",
    ]
'''

if use_trt:
    if precision_mode == 'INT8':
        converter = trt.TrtGraphConverter(
            input_saved_model_dir=output_saved_model_dir, # For frozen graphs, you need to pass in input_graph_def and nodes_blacklist parameters. nodes_blacklist is a list of output nodes.
            precision_mode=trt.TrtPrecisionMode.INT8,
            is_dynamic_op=True,
            use_calibration=True)

        frozen_graph = converter.convert()

        converted_graph_def = converter.calibrate(
            fetch_names=['detection_boxes_l1:0','detection_scores_l1:0','detection_classes_l1:0','detection_boxes_l2:0','detection_scores_l2:0','detection_classes_l2:0'],
            num_runs=10,
            feed_dict_fn=lambda: {'ved/level1/import/image_tensor:0': np.random.normal(size=(1, 1200, 1200, 3))},)
            # input_map_fn=input_map)
    else:
        converter = trt.TrtGraphConverter(
            input_saved_model_dir=output_saved_model_dir,
            precision_mode=trt.TrtPrecisionMode.FP16,
            is_dynamic_op=True,
            max_workspace_size_bytes=(1<<32), # DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES = 1 << 30
            maximum_cached_engiens=100) # DEFAULT = 1
        frozen_graph = converter.convert()
    converter.save(output_saved_model_dir_rt)

# how to extract the TensorRT calibration table after the calibration is done:

# for n in trt_graph.node:
#   if n.op == "TRTEngineOp":
#     print("Node: %s, %s" % (n.op, n.name.replace("/", "_")))
#     with tf.gfile.GFile("%s.calib_table" % (n.name.replace("/", "_")), 'wb') as f:
#       f.write(n.attr["calibration_data"].s)
