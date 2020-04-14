##################################
# author: vedanshu
#
# Will only make tfrecord of jpeg images.
##################################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import cv2
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from IPython import embed
import tensorflow as tf
from object_detection.utils import dataset_util
import random
from pathlib import Path

BASE_PATH = "/home/deploy/ved/"
json_path = BASE_PATH+"potato_l1_l2_old_18_19_20_val_gid.json"
out_path = BASE_PATH+"out"
writer = tf.python_io.TFRecordWriter(BASE_PATH+"potato_l1_l2_old_18_19_20_val_gid.record")
debug_output_img = True
mix_img_out = os.path.join(out_path, "mix_img")
Path(mix_img_out).mkdir(parents=True, exist_ok=True)
target_size=150
ncols=5
label_map = BASE_PATH+"label_map.pbtxt"
write_label_map = True
get_ar_as = True
no_ar_as = 7
l1_mask = False # L1 in grid will be cropped mask images
self_balance_category_l2 = False
if self_balance_category_l2:
    mixup_lambd = 0.6

# category_l1 = ['tomato', 'onion', 'potato',]
# category_l2 = [ ['spots holes', 'tip', 'cuts cracks', 'shrivelled', 'glare', 'back tip', 'stalk', 'green area', 'non black spots', "rotten", "water layer", "water mark", "deformed double", "unripe", "normal" ], 
#      ['smut effected', 'without skin', 'sprouting', 'rotten', 'half cut', 'tip', 'neck', 'sun burn', "glare", "double onion", "open neck", "blurred" ], 
#      ['big crack and cuts', 'decayed', 'sprouted', 'major hole', 'shriveled', 'greening', 'small crack and cuts', "badly deformed", "eyes", "confusing", "dried sprout"] ]

category_l1 = ["potato"]
category_l2 = ["decayed", "sprouted", "shriveled", "eyes", "big_crack_and_cuts", "major_hole", "dried_sprout"]

use_l2_mapping = True
if use_l2_mapping:
    # l2_mapping = {"decayed": "defect", "sprouted": "defect", "shriveled": "defect", "eyes": "defect", "big_crack_and_cuts": "defect", "major_hole": "defect", "dried_sprout": "defect"}
    l2_mapping = {"decayed": "decayed", "sprouted": "hole", "shriveled": "decayed", "eyes": "hole", "big_crack_and_cuts": "crack_and_cuts", "major_hole": "hole", "dried_sprout": "hole"}
    category_index = {k:v for k,v in zip(range(1,len(l2_mapping)+1), set(l2_mapping.values()))}
    if self_balance_category_l2:
        roi_category_l2_node_dic = {v:[] for v in set(l2_mapping.values())}

else:
    category_index = {k:v for k,v in zip(range(1,len(category_l2)+1), category_l2)}
    if self_balance_category_l2:
        roi_category_l2_node_dic = {v:[] for v in category_l2}

only_2_cat = False # only "defective" and "normal"
defective_index = [2, 4, 5]

with open(json_path, 'r') as f:
    data = json.load(f)

if write_label_map:
    with open(label_map, "w") as f_pbtxt:
        if only_2_cat:
            f_pbtxt.write("item{\n\tid: 1\n\tname: 'defective'\n}\n")
            f_pbtxt.write("item{\n\tid: 2\n\tname: 'normal'\n}\n")
        else:
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


def imgCrop(img, pts, get_mask):
    # Source: https://stackoverflow.com/a/48301735/4582711

    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()

    if get_mask:
        pts = pts - pts.min(axis=0)

        mask = np.zeros(croped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

        dst = cv2.bitwise_and(croped, croped, mask=mask)
        return dst, x, y, w, h
    return croped, x, y, w, h

def getROI(img, pts):
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    dst = cv2.bitwise_and(img, img, mask=mask)
    return dst

def createMask(height, width, pts):
    mask = np.zeros((height, width), np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    return mask

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
        self.mask = []

hierarchy_dict = {}
ignore_filename = []
for key in data:
    img = cv2.imread(os.path.join(BASE_PATH+"dataset", data[key]["filename"]))
    if checkJPG(os.path.join(BASE_PATH+"dataset", data[key]["filename"])):
        for j in range(len(data[key]["regions"])):
            if "parent_id" not in data[key]["regions"][j]["region_attributes"]:
                x_lst = np.asarray(data[key]["regions"][j]["shape_attributes"]["all_points_x"])
                y_lst = np.asarray(data[key]["regions"][j]["shape_attributes"]["all_points_y"])
                pts = [[x,y] for x,y in zip(x_lst, y_lst)]
                _img, _x1, _y1, _w1, _h1 = imgCrop(img, np.array(pts), l1_mask)
                
                if type(_img) == type(None):
                    print('...Could not read image...')
                    continue
                if _img.size < 3000:    # a little higher than 30*30*3
                    print("...ONE IMAGE SKIPPED... Too small... (",data[key]["filename"] ,") size:", _img.size)
                    continue

                _img, _x2, _y2, _scale2 = getSquareImage(_img, target_size)
                try:
                    encoded_jpg = cv2.imencode('.jpg', _img)[1].tostring()
                except:
                    print("...ONE IMAGE SKIPPED... Unable to encode...")
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

                encoded_img = node.encoded_jpg
                nparr = np.fromstring(encoded_img, np.uint8) 
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Adjusting for image cropping
                x_lst = x_lst - node.x1
                y_lst = y_lst - node.y1

                # Adjusting for image resizing
                x_lst = x_lst*node.scale2 + node.x2
                y_lst = y_lst*node.scale2 + node.y2

                xmin = np.amin(x_lst)
                xmax = np.amax(x_lst)
                ymin = np.amin(y_lst)
                ymax = np.amax(y_lst)

                pts = np.asarray([[int(x), int(y)] for x,y in zip(x_lst, y_lst)])

                if xmin / node.width < 0 or xmax / node.width < 0 or ymin / node.height < 0 or ymax / node.height < 0:
                    print("...ONE CONTOUR SKIPPED...")
                    continue
                if xmin / node.width > 1 or xmax / node.width > 1 or ymin / node.height > 1 or ymax / node.height > 1:
                    print("...ONE CONTOUR SKIPPED...")
                    continue

                try:
                    mask = createMask(node.height, node.width, pts)
                    _roi = getROI(img, np.array(pts))
                except:
                    print("Invalid mask...")
                    # embed()
                    continue
                
                txt = data[key]["regions"][j]["region_attributes"][category_l1[0]]

                try:
                    _cat_l2 = [v for v in category_l2 if v == txt][0]
                    # index = [k for k,v in category_index.items() if v == txt][0]
                except:
                    print("one tagged catogory not in list... ignoring: ", txt)
                    continue
                
                if use_l2_mapping:
                    try:
                        _mapped_value = [v for k,v in l2_mapping.items() if k == _cat_l2][0]
                        index = [k for k,v in category_index.items() if v == _mapped_value][0]
                        txt = category_index[index]
                    except:
                        print("Mapping and category_l2 did not match...")
                        continue
                else:
                    index = [k for k,v in category_index.items() if v == txt][0]

                node.xmin.append(xmin / node.width)
                node.xmax.append(xmax / node.width)
                node.ymin.append(ymin / node.height)
                node.ymax.append(ymax / node.height) 
                node.mask.append(mask)

                if self_balance_category_l2:
                    node_roi = Node()
                    node_roi.xmin.append(xmin / node.width)
                    node_roi.xmax.append(xmax / node.width)
                    node_roi.ymin.append(ymin / node.height)
                    node_roi.ymax.append(ymax / node.height) 
                    node_roi.mask.append(mask)
                    try:
                        encoded_jpg = cv2.imencode('.jpg', _roi)[1].tostring()
                    except:
                        print("...ONE IMAGE SKIPPED... Unable to encode...")
                        continue
                    node_roi.encoded_jpg = encoded_jpg
                    if only_2_cat:
                        if index in defective_index:
                            node_roi.classes_text.append(tf.compat.as_bytes("defective"))
                            node_roi.classes.append(1)
                            roi_category_l2_node_dic["defective"].append(node_roi)
                        else:
                            node_roi.classes_text.append(tf.compat.as_bytes("normal"))
                            node_roi.classes.append(2)
                            roi_category_l2_node_dic["normal"].append(node_roi)
                    else:
                        node_roi.classes_text.append(tf.compat.as_bytes(txt))
                        node_roi.classes.append(index)
                        roi_category_l2_node_dic[txt].append(node_roi)


                if only_2_cat:
                    if index in defective_index:
                        node.classes_text.append(tf.compat.as_bytes("defective"))
                        node.classes.append(1)
                    else:
                        node.classes_text.append(tf.compat.as_bytes("normal"))
                        node.classes.append(2)
                else:
                    node.classes_text.append(tf.compat.as_bytes(txt))
                    node.classes.append(index)

normal_l1_lst_id = [_id for _id, node in hierarchy_dict.items() if len(node.xmin) == 0]
if self_balance_category_l2:
    if len(normal_l1_lst_id) == 0:
        print("Unable to find l1 without defects...")
        exit(0)
    dominating_category_len = 0
    diff_dic = {}
    for k,v in roi_category_l2_node_dic.items():
        if len(v) > dominating_category_len:
            dominating_category_len = len(v)
    for k,v in roi_category_l2_node_dic.items():
        diff_dic[k] = dominating_category_len - len(v)
    start_id = max(hierarchy_dict.keys())
    _start_id = start_id

    for k,v in diff_dic.items():
        if v > 0:
            for _id in range(start_id+1,start_id+v+1):
                node_l1_id = random.choice(normal_l1_lst_id)
                node_l1 = hierarchy_dict[node_l1_id]
                encoded_jpg = node_l1.encoded_jpg
                nparr = np.fromstring(encoded_jpg, np.uint8) 
                img_l1 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                node_l2 = random.choice(roi_category_l2_node_dic[k])
                encoded_jpg = node_l2.encoded_jpg
                nparr = np.fromstring(encoded_jpg, np.uint8) 
                img_l2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                lambd = max(0, min(1, mixup_lambd))
                height = max(img_l1.shape[0], img_l2.shape[0])
                width = max(img_l1.shape[1], img_l2.shape[1])
                mix_img = np.zeros((height, width, 3), dtype='float32')
                mix_img[:img_l1.shape[0], :img_l1.shape[1], :] = img_l1.astype('float32') * lambd
                mix_img[:img_l2.shape[0], :img_l2.shape[1], :] += img_l2.astype('float32') * (1. - lambd)
                mix_img = mix_img.astype('uint8')

                mix_node = Node()
                encoded_jpg = cv2.imencode('.jpg', mix_img)[1].tostring()
                mix_node.encoded_jpg = encoded_jpg
                mix_node.width = width
                mix_node.height = height
                mix_node.image_format = b'jpg'
                mix_node.filename = tf.compat.as_bytes(str(_id))
                mix_node.xmin = node_l1.xmin + node_l2.xmin
                mix_node.xmax = node_l1.xmax + node_l2.xmax
                mix_node.ymin = node_l1.ymin + node_l2.ymin
                mix_node.ymax = node_l1.ymax + node_l2.ymax
                mix_node.classes_text = node_l1.classes_text + node_l2.classes_text
                mix_node.classes = node_l1.classes + node_l2.classes
                mix_node.mask = node_l1.mask + node_l2.mask

                hierarchy_dict[_id] = mix_node
            start_id = start_id+v
    if debug_output_img:
        for _id in range(_start_id+1, max(hierarchy_dict.keys()) + 1):
            node = hierarchy_dict[_id]
            encoded_jpg = node.encoded_jpg
            nparr = np.fromstring(encoded_jpg, np.uint8) 
            _img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            for xmin, ymin, xmax, ymax, txt, _mask in zip(node.xmin, node.ymin, node.xmax, node.ymax, node.classes_text, node.mask):
                cv2.rectangle(_img, (int(xmin*node.width), int(ymin*node.height)), (int(xmax*node.width), int(ymax*node.height)), (255,0,0), 2)
                cv2.putText(_img, str(txt), (int(xmin*node.width), int(ymin*node.height)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
                _bgd_mask = cv2.bitwise_not(_mask)
                _bgd = cv2.bitwise_and(_img,_img,mask = _bgd_mask)
                roi = cv2.bitwise_and(_img,_img,mask = _mask)
                _roi = roi*[0,0,0.6]
                _img = _bgd + _roi
            
            _path = os.path.join(mix_img_out, str(_id) +"_ann.jpg")
            # embed()
            cv2.imwrite(_path, _img)

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

lst_xmins = []
lst_xmaxs = []
lst_ymins = []
lst_ymaxs = []
lst_width = []
lst_height = []
lst_classes = []

while not_ended:
    grid_img = np.zeros((ncols*ncols,target_size,target_size,3))
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes_text = []
    classes = []
    encoded_jpg_mask_lst = []

    rand_size = random.randint(1,ncols*ncols)
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

        for xmin, ymin, xmax, ymax, txt, _index, _mask in zip(node.xmin, node.ymin, node.xmax, node.ymax, node.classes_text, node.classes, node.mask):
            _x, _y = getShiftingCord(i, ncols)

            grid_img_mask = np.zeros((normalizer,normalizer))
            grid_img_mask[_y:(_y+_mask.shape[0]), _x:(_x+_mask.shape[1])] = _mask
            grid_img_mask = cv2.imencode('.png', grid_img_mask)[1].tostring()
            encoded_jpg_mask_lst.append(grid_img_mask)
            
            xmins.append( (xmin*node.width + _x)/normalizer)
            ymins.append( (ymin*node.height + _y)/normalizer)
            xmaxs.append( (xmax*node.width + _x)/normalizer)
            ymaxs.append( (ymax*node.height + _y)/normalizer)
            classes_text.append(txt)
            classes.append(_index)

            lst_xmins.append((xmin*node.width + _x))
            lst_xmaxs.append((xmax*node.height + _x))
            lst_ymins.append((ymin*node.width + _y))
            lst_ymaxs.append((ymax*node.height + _y))
            lst_width.append(normalizer)
            lst_height.append(normalizer)
            lst_classes.append(_index)

    grid_img = gallery(grid_img, ncols)
    encoded_jpg = cv2.imencode('.jpg', grid_img)[1].tostring()

    if debug_output_img:
        for xmin, ymin, xmax, ymax, txt, _mask in zip(xmins, ymins, xmaxs, ymaxs, classes_text, encoded_jpg_mask_lst):
            cv2.rectangle(grid_img, (int(xmin*normalizer), int(ymin*normalizer)), (int(xmax*normalizer), int(ymax*normalizer)), (255,0,0), 2)
            cv2.putText(grid_img, str(txt), (int(xmin*normalizer), int(ymin*normalizer)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
            nparr = np.fromstring(_mask, np.uint8) 
            mask_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            _bgd_mask = cv2.bitwise_not(mask_np)
            _bgd = cv2.bitwise_and(grid_img,grid_img,mask = _bgd_mask)
            roi = cv2.bitwise_and(grid_img,grid_img,mask = mask_np)
            _roi = roi*[0,0,0.6]
            grid_img = _bgd + _roi
        if not_ended:
            _path = os.path.join(out_path, str(lst_id[start_index+i]) +"_ann.jpg")
            cv2.imwrite(_path, grid_img)
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
    'image/object/mask': dataset_util.bytes_list_feature(encoded_jpg_mask_lst)
    }))
    writer.write(tf_example.SerializeToString())
    start_index = start_index + i + 1

if get_ar_as:
    data = pd.DataFrame({'xmins': lst_xmins, 'ymins': lst_ymins, 'xmaxs': lst_xmaxs, 'ymaxs': lst_ymaxs, 'width': lst_width, 'height': lst_height})
    data['w'] = data['xmaxs'] - data['xmins']
    data['h'] = data['ymaxs'] - data['ymins']

    data['b_w'] = target_size*data['w']/data['width']
    data['b_h'] = target_size*data['h']/data['height']

    X = data.as_matrix(columns=['b_w', 'b_h'])
    K = KMeans(no_ar_as, random_state=0)
    labels = K.fit(X)

    out = labels.cluster_centers_

    ar = out[:,0]/out[:,1]
    scale = out[:,1]*np.sqrt(ar)/256

    print("Aspect Ratios: ",ar)

    print("Scales: ", scale)

writer.close()

