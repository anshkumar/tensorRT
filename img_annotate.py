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

json_path = "/Users/vedanshu/tfrecord/train.json"
out_path = "/Users/vedanshu/tfrecord/out"
commodity_lst = ["tomato"]
writer = tf.python_io.TFRecordWriter("train.record")
target_size=1200
label_map = "/Users/vedanshu/tfrecord/label_map.pbtxt"
write_label_map = False

# category = ['tomato', 'tomato spots holes', 'tomato tip', 'tomato cuts cracks', 'tomato shrivelled', 'tomato glare', 'tomato back tip', 'tomato stalk', 'tomato green area', 'tomato non black spots', "tomato rotten", "tomato water layer", "tomato water mark", "tomato deformed double", "tomato unripe", "tomato normal"] 
    # 'onion', 'onion smut effected', 'onion without skin', 'onion sprouting', 'onion rotten', 'onion half cut', 'onion tip', 'onion neck', 'onion sun burn', "onion glare", "onion double onion", "onion open neck", "onion blurred", 
    # 'potato', 'potato big crack and cuts', 'potato decayed', 'potato sprouted', 'potato major hole', 'potato shriveled', 'potato greening', 'potato small crack and cuts', "potato badly deformed", "potato eyes", "potato confusing", "potato dried sprout"]
category = ['tomato', 'spots holes', 'tip', 'cuts cracks', 'shrivelled', 'glare', 'back tip', 'stalk', 'green area']
# category_weights = [1.0, 25.99502487562189, 9.816815406294035, 78.27715355805243, 108.29015544041451, 5.9971305595408895, 152.55474452554745, 175.63025210084032, 34.26229508196721]
category_weights = [0.00569377990430622, 0.1480099502487562, 0.055894786284640674, 0.44569288389513106, 0.616580310880829, 0.03414634146341464, 0.8686131386861314, 1.0, 0.19508196721311474]
category_index = {k:v for k,v in zip(range(1,len(category)+1), category)}
category_weights_index = {k:v for k,v in zip(category, category_weights)}

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
    _img = cv2.imread(os.path.join("/Users/vedanshu/tfrecord/dataset", data[key]["filename"]))
    if checkJPG(os.path.join("/Users/vedanshu/tfrecord/dataset", data[key]["filename"])):
        # _img, _x, _y, _scale = getSquareImage(img, target_size)
        encoded_jpg = cv2.imencode('.jpg', _img)[1].tostring()
        filename = tf.compat.as_bytes(data[key]["filename"])
        image_format = b'jpg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []
        weights = []
        width = _img.shape[1]
        height = _img.shape[0]

        for j in range(len(data[key]["regions"])):
            x_lst = np.asarray(data[key]["regions"][j]["shape_attributes"]["all_points_x"])
            y_lst = np.asarray(data[key]["regions"][j]["shape_attributes"]["all_points_y"])
            pts = [[x,y] for x,y in zip(x_lst, y_lst)]
            x,y,w,h = cv2.boundingRect(np.array(pts))
            rect_area = w*h
            if rect_area < 300:
                print("Area too less ...ONE CONTOUR SKIPPED...")
                continue
            # xmin = int(np.amin(x_lst)*_scale) + _x
            # xmax = int(np.amax(x_lst)*_scale) + _x
            # ymin = int(np.amin(y_lst)*_scale) + _y
            # ymax = int(np.amax(y_lst)*_scale) + _y

            xmin = int(np.amin(x_lst))
            xmax = int(np.amax(x_lst))
            ymin = int(np.amin(y_lst))
            ymax = int(np.amax(y_lst))
            
            cv2.rectangle(_img, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
            try:
                _commodity = [i for i in commodity_lst if i in data[key]["regions"][j]["region_attributes"]][0]
            except:
                print("commodity name not in list")
                embed()
            if _commodity == data[key]["regions"][j]["region_attributes"][_commodity]:
                text = _commodity
            else:
                # text = _commodity + " " + data[key]["regions"][j]["region_attributes"][_commodity]
                text = data[key]["regions"][j]["region_attributes"][_commodity]
            # embed()
            try:
                index = [k for k,v in category_index.items() if v == text.strip()][0]
            except:
                print("one tagged catogory not in list... ignoring: ", text)
                continue
            xmins.append( xmin / width)
            xmaxs.append( xmax / width)
            ymins.append( ymin / height)
            ymaxs.append( ymax / height)
            classes_text.append(tf.compat.as_bytes(text))
            classes.append(index)
            weights.append(category_weights_index[text])
        #     cv2.putText(_img, text + str(category_weights_index[text]), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
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
        'image/object/weight': dataset_util.float_list_feature(weights)
        }))
        writer.write(tf_example.SerializeToString())
        
writer.close()


