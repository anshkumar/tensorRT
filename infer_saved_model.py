import tensorflow as tf
import cv2
import numpy as np

IMAGE_PATH = "/Users/vedanshu/test.jpg"
image = cv2.imread(IMAGE_PATH)

input_names = ['image_tensor']

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.5, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]),
                  (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale,
                (255, 255, 255), thickness)

detection_graph1 = tf.Graph()

with detection_graph1.as_default():
    tf_sess1 = tf.Session(graph=detection_graph1)
    model = tf.saved_model.loader.load(tf_sess1, ["serve"], "/Users/vedanshu/ckpt/saved_model")
    
tf_input = tf_sess1.graph.get_tensor_by_name(input_names[0] + ':0')
tf_scores = tf_sess1.graph.get_tensor_by_name('detection_scores:0')
tf_boxes = tf_sess1.graph.get_tensor_by_name('detection_boxes:0')
tf_classes = tf_sess1.graph.get_tensor_by_name('detection_classes:0')
tf_num_detections = tf_sess1.graph.get_tensor_by_name('num_detections:0')
scores, boxes, classes, num_detections = tf_sess1.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={tf_input: image[None, ...]})

boxes = boxes[0]  # index by 0 to remove batch dimension
scores = scores[0]
classes = classes[0]
num_detections = int(num_detections[0])

boxes_pixels = []
for i in range(num_detections):
    # scale box to image coordinates
    box = boxes[i] * np.array([image.shape[0],
                               image.shape[1], image.shape[0], image.shape[1]])
    box = np.round(box).astype(int)
    boxes_pixels.append(box)

boxes_pixels = np.array(boxes_pixels)

for i in range(num_detections):
    box = boxes_pixels[i]
    box = np.round(box).astype(int)
    image = cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
    label = "{}:{:.2f}".format(int(classes[i]), scores[i])
    draw_label(image, (box[1], box[0]), label)

cv2.imwrite("/Users/vedanshu/out.jpg", image)


