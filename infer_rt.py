#############################
# author: Vedanshu
#############################

import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import numpy as np
import cv2, queue, threading, time
import time
from IPython import embed

input_names = ['image_tensor']
pb_fname1 = "/home/xavier/data/trt_graph.pb"
pb_fname2 = "/home/xavier/data/trt_graph.pb.bck"

class_lst = ['tomato', 'tomato spots holes', 'tomato tip', 'tomato cuts cracks', 'tomato shrivelled', 'tomato glare', 'tomato back tip', 'tomato stalk', 'tomato green area', 'tomato non black spots', "tomato rotten", "tomato water layer", "tomato water mark", "tomato deformed double", "tomato unripe", "tomato normal", 
    'onion', 'onion smut effected', 'onion without skin', 'onion sprouting', 'onion rotten', 'onion half cut', 'onion tip', 'onion neck', 'onion sun burn', "onion glare", "onion double onion", "onion open neck", "onion blurred", 
    'potato', 'potato big crack and cuts', 'potato decayed', 'potato sprouted', 'potato major hole', 'potato shriveled', 'potato greening', 'potato small crack and cuts', "potato badly deformed", "potato eyes", "potato confusing", "potato dried sprout"]

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except Queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()
  
  def release(self):
    self.cap.release()

vcap = VideoCapture('http://10.42.0.211:8080/?action=stream')
trt_graph1 = get_frozen_graph(pb_fname1)
trt_graph2 = get_frozen_graph(pb_fname2)

detection_graph1 = tf.Graph()

with detection_graph1.as_default():
    tf.import_graph_def(trt_graph1, name='')
    tf_sess1 = tf.Session(graph=detection_graph1)

detection_graph2 = tf.Graph()

with detection_graph2.as_default():
    tf.import_graph_def(trt_graph2, name='')
    tf_sess2 = tf.Session(graph=detection_graph2)

tf_input1 = tf_sess1.graph.get_tensor_by_name(input_names[0] + ':0')
tf_scores1 = tf_sess1.graph.get_tensor_by_name('detection_scores:0')
tf_boxes1 = tf_sess1.graph.get_tensor_by_name('detection_boxes:0')
tf_classes1 = tf_sess1.graph.get_tensor_by_name('detection_classes:0')
tf_num_detections1 = tf_sess1.graph.get_tensor_by_name('num_detections:0')
tf_detection_masks1 = tf_sess1.graph.get_tensor_by_name('detection_masks:0')

tf_input2 = tf_sess2.graph.get_tensor_by_name(input_names[0] + ':0')
tf_scores2 = tf_sess2.graph.get_tensor_by_name('detection_scores:0')
tf_boxes2 = tf_sess2.graph.get_tensor_by_name('detection_boxes:0')
tf_classes2 = tf_sess2.graph.get_tensor_by_name('detection_classes:0')
tf_num_detections2 = tf_sess2.graph.get_tensor_by_name('num_detections:0')

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.5, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]),
                  (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale,
                (255, 255, 255), thickness)

def non_max_suppression(boxes, probs=None, nms_threshold=0.3):
    """Non-max suppression

    Arguments:
        boxes {np.array} -- a Numpy list of boxes, each one are [x1, y1, x2, y2]
    Keyword arguments
        probs {np.array} -- Probabilities associated with each box. (default: {None})
        nms_threshold {float} -- Overlapping threshold 0~1. (default: {0.3})

    Returns:
        list -- A list of selected box indexes.
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > nms_threshold)[0])))
    # return only the bounding boxes indexes
    return pick

fps = 0.0
full_scrn = False
WINDOW_NAME = "IntelloLabs"
tic = time.time()
    
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while(True):
    # Capture frame-by-frame
    frame = vcap.read()
    #print cap.isOpened(), ret
    if frame is not None:
        # Display the resulting frame
        scores, boxes, classes, num_detections, masks = tf_sess1.run([tf_scores1, tf_boxes1, tf_classes1, tf_num_detections1, tf_detection_masks1], feed_dict={tf_input1: frame[None, ...]})
        boxes = boxes[0]  # index by 0 to remove batch dimension
        scores = scores[0]
        classes = classes[0]
        num_detections = int(num_detections[0])
        masks = masks[0]

        # Boxes unit in pixels (image coordinates).
        boxes_pixels = []
        for i in range(num_detections):
            # scale box to image coordinates
            box = boxes[i] * np.array([frame.shape[0],
                               frame.shape[1], frame.shape[0], frame.shape[1]])
            box = np.round(box).astype(int)
            boxes_pixels.append(box)
        boxes_pixels = np.array(boxes_pixels)

        # Remove overlapping boxes with non-max suppression, return picked indexes.
        pick = non_max_suppression(boxes_pixels, scores[:num_detections], 0.5)
        # print(pick)
        clone = frame.copy()
        for i in pick:
            if scores[i] > 0.5:
                box = boxes_pixels[i]
                box = np.round(box).astype(int)
                (startY, startX, endY , endX) = box.astype("int")
                boxW = endX - startX
                boxH = endY - startY
                mask = masks[i]
                mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_NEAREST)
                # mask = (mask > 0.4)
                roi = clone[startY:endY, startX:endX]
                visMask = (mask * 255).astype("uint8")
                instance = cv2.bitwise_and(roi, roi, mask=visMask)

                # cv2.imshow("ROI", roi)
                # cv2.imshow("Mask", visMask)
                # cv2.imshow("Segmented", instance)
                # embed()

                _scores, _boxes, _classes, _num_detections = tf_sess2.run([tf_scores2, tf_boxes2, tf_classes2, tf_num_detections2], feed_dict={tf_input2: instance[None, ...]})
                _boxes = _boxes[0]  # index by 0 to remove batch dimension
                _scores = _scores[0]
                _classes = _classes[0]
                _num_detections = int(_num_detections[0])

                _boxes_pixels = []
                for i in range(_num_detections):
                    # scale box to image coordinates
                    _box = _boxes[i] * np.array([instance.shape[0],
                                       instance.shape[1], instance.shape[0], instance.shape[1]])
                    _box = np.round(_box).astype(int)
                    _boxes_pixels.append(_box)
                _boxes_pixels = np.array(_boxes_pixels)

                _pick = non_max_suppression(_boxes_pixels, scores[:_num_detections], 0.5)
                for i in _pick:
                    if _scores[i] > 0.5:
                        _box = _boxes_pixels[i]
                        _box = np.round(_box).astype(int)
                        image = cv2.rectangle(frame, (_box[1] + startX, _box[0] + startY), (_box[3] + startX, _box[2] + startY), (0, 255, 0), 2)
                        _label = "{}:{:.2f}".format(int(_classes[i]), _scores[i])
                        draw_label(image, (_box[1] + startX, _box[0] + startY), _label)
                
                # Draw bounding box.
                image = cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
                #embed()
                label = "{}:{:.2f}".format(class_lst[int(classes[i])+1], scores[i])
                # Draw label (class index and probability).
                draw_label(image, (box[1], box[0]), label)

        cv2.imshow(WINDOW_NAME,image)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.9 + curr_fps*0.1)
        tic = toc
        print("fps = ", fps)
        # Press q to close the video windows before it ends if you want
        if cv2.waitKey(22) & 0xFF == ord('q'):
            break
    else:
        print("Frame is None")
        break

# When everything done, release the capture
vcap.release()
cv2.destroyAllWindows()
print("Video stop")

