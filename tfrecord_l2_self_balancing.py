##################################
# author: vedanshu
#
# Will only make tfrecord of jpeg images.
##################################

import cv2
import os
import json
import numpy as np
from IPython import embed
import tensorflow as tf
from object_detection.utils import dataset_util
import random

json_path = "/Users/vedanshu/tfrecord/train_tomato.json"
out_path = "/Users/vedanshu/tfrecord/out"
writer = tf.python_io.TFRecordWriter("train.record")
debug_output_img = True
target_size=150
ncols=5
label_map = "/Users/vedanshu/tfrecord/label_map.pbtxt"
write_label_map = False

# category_l1 = ['tomato', 'onion', 'potato',]
# category_l2 = [ ['spots holes', 'tip', 'cuts cracks', 'shrivelled', 'glare', 'back tip', 'stalk', 'green area', 'non black spots', "rotten", "water layer", "water mark", "deformed double", "unripe", "normal" ], 
#      ['smut effected', 'without skin', 'sprouting', 'rotten', 'half cut', 'tip', 'neck', 'sun burn', "glare", "double onion", "open neck", "blurred" ], 
#      ['big crack and cuts', 'decayed', 'sprouted', 'major hole', 'shriveled', 'greening', 'small crack and cuts', "badly deformed", "eyes", "confusing", "dried sprout"] ]

category_l1 = ['tomato']
category_l2 = ['tomato', 'spots_holes', 'tip', 'cuts_cracks', 'shrivelled', 'glare', 'back_tip', 'stalk', 'green_area']

category_index = {k:v for k,v in zip(range(1,len(category_l2)+1), category_l2)}
defective_index = [2, 4, 5, 9]

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

    square[y:y+height, x:x+width] = cv2.resize( img , (width, height) )

    return square, x, y, scale

def imgCrop(img, pts):
    # Source: https://stackoverflow.com/a/48301735/4582711

    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()

    pts = pts - pts.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    dst = cv2.bitwise_and(croped, croped, mask=mask)
    return dst, x, y, w, h

class Node: 
    def __init__(self):
        self.encoded_jpg = None
        self.filename = None
        self.image_format = None
        self.width = None
        self.height = None
        self.x1 = None
        self.y1 = None
        self.x2 = None
        self.y2 = None
        self.scale2 = None
        self.xmin = []
        self.xmax = []
        self.ymin = []
        self.ymax = []
        self.classes_text = []
        self.classes = []

hierarchy_dict = {}
ignore_filename = []
for key in data:
    img = cv2.imread(os.path.join("/Users/vedanshu/tfrecord/dataset", data[key]["filename"]))
    if checkJPG(os.path.join("/Users/vedanshu/tfrecord/dataset", data[key]["filename"])):
        for j in range(len(data[key]["regions"])):
            if "parent_id" not in data[key]["regions"][j]["region_attributes"]:
                x_lst = np.asarray(data[key]["regions"][j]["shape_attributes"]["all_points_x"])
                y_lst = np.asarray(data[key]["regions"][j]["shape_attributes"]["all_points_y"])
                pts = [[x,y] for x,y in zip(x_lst, y_lst)]
                _img, _x1, _y1, _w1, _h1 = imgCrop(img, np.array(pts))
                
                if type(_img) == type(None):
                    print('...Could not read image...')
                    continue
                if _img.size < 10000:    # a little higher than 50*50*3
                    print("...ONE IMAGE SKIPPED...")
                    continue

                _img, _x2, _y2, _scale2 = getSquareImage(_img, target_size)
                try:
                    encoded_jpg = cv2.imencode('.jpg', _img)[1].tostring()
                except:
                    print("...ONE IMAGE SKIPPED...")
                    continue
                _id = data[key]["regions"][j]["region_attributes"]["id"]
                node = Node()
                node.encoded_jpg = encoded_jpg
                node.filename = tf.compat.as_bytes(data[key]["filename"])
                node.image_format = b'jpg'
                node.width = _img.shape[1]
                node.height = _img.shape[0]
                node.x1 = _x1
                node.y1 = _y1
                node.x2 = _x2
                node.y2 = _y2
                node.scale2 = _scale2
                hierarchy_dict[_id] = node
    else:
        ignore_filename.append(data[key]["filename"])

for key in data:
    if data[key]["filename"] not in ignore_filename:
        for j in range(len(data[key]["regions"])):
            if "parent_id" in data[key]["regions"][j]["region_attributes"] and data[key]["regions"][j]["region_attributes"]["parent_id"] in hierarchy_dict:
                _id = data[key]["regions"][j]["region_attributes"]["parent_id"]
                x_lst = np.asarray(data[key]["regions"][j]["shape_attributes"]["all_points_x"])
                y_lst = np.asarray(data[key]["regions"][j]["shape_attributes"]["all_points_y"])
                node = hierarchy_dict[_id]

                # Adjusting for image cropping
                xmin = np.amin(x_lst) - node.x1
                xmax = np.amax(x_lst) - node.x1
                ymin = np.amin(y_lst) - node.y1
                ymax = np.amax(y_lst) - node.y1

                # Adjusting for image resizing
                xmin = xmin*node.scale2 + node.x2
                xmax = xmax*node.scale2 + node.x2
                ymin = ymin*node.scale2 + node.y2
                ymax = ymax*node.scale2 + node.y2

                if xmin / node.width < 0 or xmax / node.width < 0 or ymin / node.height < 0 or ymax / node.height < 0:
                    print("...ONE CONTOUR SKIPPED...")
                    continue
                if xmin / node.width > 1 or xmax / node.width > 1 or ymin / node.height > 1 or ymax / node.height > 1:
                    print("...ONE CONTOUR SKIPPED...")
                    continue
                
                txt = data[key]["regions"][j]["region_attributes"][category_l1[0]]

                try:
                    index = [k for k,v in category_index.items() if v == txt][0]
                except:
                    print("one tagged catogory not in list... ignoring: ", txt)
                    continue
                
                node.xmin.append(xmin / node.width)
                node.xmax.append(xmax / node.width)
                node.ymin.append(ymin / node.height)
                node.ymax.append(ymax / node.height) 

                if index in defective_index:
                    node.classes_text.append(tf.compat.as_bytes("defective"))
                    node.classes.append(1)
                else:
                    node.classes_text.append(tf.compat.as_bytes("normal"))
                    node.classes.append(2)

def gallery(array, ncols=5):
    # https://stackoverflow.com/a/42041135/4582711
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result

def getShiftingCord(index, ncols=5):
    return (index%ncols)*target_size, int(index/ncols)*target_size


grid_img_lst =[]
lst_id = [_id for _id, node in hierarchy_dict.items() if len(node.xmin) > 0]
not_ended = True
start_index = 0

while not_ended:
    grid_img = np.zeros((25,150,150,3))
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes_text = []
    classes = []

    rand_size = random.randint(1,25)
    normalizer = ncols*target_size
    
    for i in range(rand_size):
        try:
            node = hierarchy_dict[lst_id[start_index+i]]
            _img = node.encoded_jpg
        except:
            print("List ended...!!!")
            not_ended = False
            break
        nparr = np.fromstring(_img, np.uint8) 
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        grid_img[i] = img_np

        for xmin, ymin, xmax, ymax, txt, _index in zip(node.xmin, node.ymin, node.xmax, node.ymax, node.classes_text, node.classes):
            _x, _y = getShiftingCord(i, ncols)
            xmins.append( (xmin*node.width + _x)/normalizer)
            ymins.append( (ymin*node.height + _y)/normalizer)
            xmaxs.append( (xmax*node.width + _x)/normalizer)
            ymaxs.append( (ymax*node.height + _y)/normalizer)
            classes_text.append(txt)
            classes.append(_index)

    grid_img = gallery(grid_img, ncols)
    encoded_jpg = cv2.imencode('.jpg', _img)[1].tostring()

    if debug_output_img:
        for xmin, ymin, xmax, ymax, txt in zip(xmins, ymins, xmaxs, ymaxs, classes_text):
            cv2.rectangle(grid_img, (int(xmin*normalizer), int(ymin*normalizer)), (int(xmax*normalizer), int(ymax*normalizer)), (255,0,0), 2)
            cv2.putText(grid_img, str(txt), (int(xmin*normalizer), int(ymin*normalizer)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

        if not_ended:
            _path = os.path.join(out_path, str(lst_id[start_index+i]) +"_ann.jpg")
            cv2.imwrite(_path, grid_img)
    else:    
        tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(normalizer),
        'image/width': dataset_util.int64_feature(normalizer),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(b'jpg'),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        writer.write(tf_example.SerializeToString())
    start_index = start_index + i + 1

writer.close()

