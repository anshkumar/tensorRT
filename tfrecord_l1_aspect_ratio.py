##################################
# author: vedanshu
##################################

import cv2
import os
import json
import numpy as np
from IPython import embed
import tensorflow as tf
from object_detection.utils import dataset_util

json_path = "/home/deploy/ved/train.json"
out_path = "/home/deploy/ved/out"
commodity_lst = ["tomato", "onion", "potato"]
writer = tf.python_io.TFRecordWriter("train.record")
target_size = 1200
label_map = "/home/deploy/ved/label_map.pbtxt"
write_label_map = True

# category = ['tomato','onion', 'potato']
category = ['tomato']

category_index = {k:v for k,v in zip(range(1,len(category)+1), category)}

with open(json_path, 'r') as f:
    data = json.load(f)

if write_label_map:
    with open(label_map, "w") as f_pbtxt:
        for k,v in category_index.items():
            f_pbtxt.write("item{{\n\tid: {}\n\tname: '{}'\n}}\n".format(k,v))

def checkJPG(fn):
    with tf.Graph().as_default():
        try:
            image_contents = tf.read_file(fn)
            image = tf.image.decode_jpeg(image_contents, channels=3)
            init_op = tf.initialize_all_tables()
            with tf.Session() as sess:
                sess.run(init_op)
                tmp = sess.run(image)
        except:
            print("Corrupted file: ", fn)
            return False
    return True

def getSquareImage( img, target_width = 500 ):
    width = img.shape[1]
    height = img.shape[0]

    square = np.zeros( (target_width, target_width, 3), dtype=np.uint8 )

    max_dim = width if width >= height else height
    scale = target_width / max_dim
    
    if ( width >= height ):
        width = target_width
        x = 0
        height = int(height * scale)
        y = int(( target_width - height ) / 2)
    else:
        y = 0
        height = target_width
        width = int(width * scale)
        x = int(( target_width - width ) / 2)
    # embed()
    square[y:y+height, x:x+width] = cv2.resize( img , (width, height) )

    return square, x, y, scale

for key in data:
    img = cv2.imread(os.path.join("/home/deploy/ved/dataset", data[key]["filename"]))
    if checkJPG(os.path.join("/home/deploy/ved/dataset", data[key]["filename"])):
        encoded_jpg = cv2.imencode('.jpg', _img)[1].tostring()
        filename = tf.compat.as_bytes(data[key]["filename"])
        image_format = b'jpg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []
        width = _img.shape[1]
        height = _img.shape[0]

        for j in range(len(data[key]["regions"])):
            x_lst = np.asarray(data[key]["regions"][j]["shape_attributes"]["all_points_x"])
            y_lst = np.asarray(data[key]["regions"][j]["shape_attributes"]["all_points_y"])
            xmin = int(np.amin(x_lst))
            xmax = int(np.amax(x_lst))
            ymin = int(np.amin(y_lst))
            ymax = int(np.amax(y_lst))
            
            try:
                _commodity = [i for i in commodity_lst if i in data[key]["regions"][j]["region_attributes"]][0]
            except:
                print("commodity name not in list")
                embed()
            if _commodity == data[key]["regions"][j]["region_attributes"][_commodity]:
                text = _commodity
            
                try:
                    index = [k for k,v in category_index.items() if v == text.strip()][0]
                except:
                    print("one tagged catogory not in list")
                    embed()
                xmins.append( xmin / width)
                xmaxs.append( xmax / width)
                ymins.append( ymin / height)
                ymaxs.append( ymax / height)
                classes_text.append(tf.compat.as_bytes(text))
                classes.append(index)
                # cv2.rectangle(_img, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
                # cv2.putText(_img, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        # _path = os.path.join(out_path, os.path.splitext(data[key]["filename"])[0]+"_ann"+os.path.splitext(data[key]["filename"])[1])
        # cv2.imwrite(_path, _img)

        tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        writer.write(tf_example.SerializeToString())
        
writer.close()


