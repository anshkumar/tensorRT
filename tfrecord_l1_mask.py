##################################
# author: vedanshu
##################################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import cv2
import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from IPython import embed
import tensorflow as tf
from object_detection.utils import dataset_util

json_path= "/home/deploy/ved/potato_17_18_19_gid.json"
out_path = "/home/deploy/ved/out"
#commodity_lst = ["tomato", "onion", "potato"]
commodity_lst = ["potato"]
writer = tf.python_io.TFRecordWriter("/home/deploy/ved/potato_l1_17_18_19_gid.record")
target_size = 600
label_map = "/home/deploy/ved/label_map_potato_l1.pbtxt"
write_label_map = True
get_ar_as = True
no_ar_as = 4

category=["potato"]
# ["full_black", "floater", "slight_insect_damage", "immature_or_unripe", "brown", "fungus_damaged", "shells", "robusta_parchment", "partial_black", "full_sour", "withered", "dried_cherry_pod", "severe_insect_damage", "partial_sour", "hull_or_husk", "robusta_cherry", ]
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

def createMask(img, pts):
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    return mask

lst_xmins = []
lst_xmaxs = []
lst_ymins = []
lst_ymaxs = []
lst_width = []
lst_height = []

for key in data:
    img = cv2.imread(os.path.join("/home/deploy/ved/dataset", data[key]["filename"]))
    if checkJPG(os.path.join("/home/deploy/ved/dataset", data[key]["filename"])):
        _img, _x, _y, _scale = getSquareImage(img, target_size)
        encoded_jpg = cv2.imencode('.jpg', _img)[1].tostring()
        filename = tf.compat.as_bytes(data[key]["filename"])
        image_format = b'jpg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []
        encoded_mask_png = []
        width = _img.shape[1]
        height = _img.shape[0]

        if type(_img) == type(None):
            print('...Could not read image...')
            continue
        if _img.size < 3000:    # a little higher than 50503
            print("...ONE IMAGE SKIPPED...")
            continue

        for j in range(len(data[key]["regions"])):
            x_lst = np.asarray(data[key]["regions"][j]["shape_attributes"]["all_points_x"])
            y_lst = np.asarray(data[key]["regions"][j]["shape_attributes"]["all_points_y"])
            pts = np.asarray([[int(x*_scale + _x), int(y*_scale + _y)] for x,y in zip(x_lst, y_lst)])

            try:
                mask = createMask(_img, pts)
            except:
                print("Invalid mask...")
                continue

            xmin = int(np.amin(x_lst)*_scale) + _x
            xmax = int(np.amax(x_lst)*_scale) + _x
            ymin = int(np.amin(y_lst)*_scale) + _y
            ymax = int(np.amax(y_lst)*_scale) + _y

            boxW = xmax - xmin
            boxH = ymax - ymin
            if boxW == 0 or boxH == 0:
                print("...ONE CONTOUR SKIPPED... (boxW | boxH) = 0")
                continue

            if boxW*boxH < 100:
                print("...ONE CONTOUR SKIPPED... (boxW*boxH) < 100")
                continue
            
            # cv2.rectangle(_img, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
            if xmin / width <= 0 or xmax / width <= 0 or ymin / height <= 0 or ymax / height <= 0:
                print("...ONE CONTOUR SKIPPED... (x | y) <= 0")
                continue
            if xmin / width >= 1 or xmax / width >= 1 or ymin / height >= 1 or ymax / height >= 1:
                print("...ONE CONTOUR SKIPPED... (x | y) >= 1")
                continue 
            try:
                _commodity = [i for i in commodity_lst if i in data[key]["regions"][j]["region_attributes"]][0]
            except:
                print("commodity name not in list: ", data[key]["regions"][j]["region_attributes"])
                continue
                # embed()
            if _commodity == data[key]["regions"][j]["region_attributes"][_commodity]:
                text = _commodity
                try:
                    index = [k for k,v in category_index.items() if v == text.strip()][0]
                except:
                    print("one tagged catogory not in list: ", text)
                    continue
                    # embed()
                xmins.append( xmin / width)
                xmaxs.append( xmax / width)
                ymins.append( ymin / height)
                ymaxs.append( ymax / height)
                classes_text.append(tf.compat.as_bytes(text))
                classes.append(index)
                encoded_png = cv2.imencode('.png', mask)[1].tostring()
                encoded_mask_png.append(encoded_png)

                lst_xmins.append(xmin)
                lst_xmaxs.append(xmax)
                lst_ymins.append(ymin)
                lst_ymaxs.append(ymax)
                lst_width.append(width)
                lst_height.append(height)
        #     cv2.putText(_img, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        # _path = os.path.join(out_path, os.path.splitext(data[key]["filename"])[0]+"_ann"+os.path.splitext(data[key]["filename"])[1])
        # cv2.imwrite(_path, _img)

        
        if len(xmins) > 0:
            tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            # 'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
            'image/object/mask': dataset_util.bytes_list_feature(encoded_mask_png)
            }))
            writer.write(tf_example.SerializeToString())
        else:
            print("len(xmins) == 0")

if get_ar_as:
    data = pd.DataFrame({'xmins': lst_xmins, 'ymins': lst_ymins, 'xmaxs': lst_xmaxs, 'ymaxs': lst_ymaxs, 'width': lst_width, 'height': lst_height})
    data['w'] = data['xmaxs'] - data['xmins']
    data['h'] = data['ymaxs'] - data['ymins']

    data['b_w'] = target_size*data['w']/data['width']
    data['b_h'] = target_size*data['h']/data['height']

    X = data[['b_w', 'b_h']].to_numpy()
    K = KMeans(no_ar_as, random_state=0)
    labels = K.fit(X)

    out = labels.cluster_centers_

    ar = out[:,0]/out[:,1]
    scale = out[:,1]*np.sqrt(ar)/256

    print("Aspect Ratios: ",ar)

    print("Scales: ", scale)

writer.close()



