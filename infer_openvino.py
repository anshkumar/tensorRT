import cv2
import numpy as np
from IPython import embed
from openvino.inference_engine import IENetwork, IECore
import json
from shapely.geometry import Polygon
import psutil
import threading
from time import sleep

def createMask(img, pts):
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    # print(mask)
    return mask

def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou

def mask_to_polygon(mask) :
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    points = []
    points_drw = []
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    clen = len(biggest_contour)
    clen -= 2
    points.append(biggest_contour[0][0].tolist())
    points_drw.append(contours[0])
    i = 1
    while i <= clen:
        points.append(biggest_contour[i][0].tolist())
        i += 1
    points.append(biggest_contour[clen + 1][0].tolist())
    return points

utilzation = []
stop = False
def get_utilization():
    global stop
    while True:
        if not stop:
            utilzation.append(psutil.cpu_percent())
            sleep(0.001)
        else:
            break
    print("threading stopped...")

t1 = threading.Thread(target=get_utilization)
t1.start() 

print("thread started...")

json_path = "data/json/json_path.json"
with open(json_path, 'r') as f:
    data = json.load(f)

for key in data:
    manual_data_array =[]
    frame = cv2.imread(os.path.join("/home/deploy/ved/dataset", data[key]["filename"]))
    frame_h, frame_w = frame.shape[:2]
    print("Processing image ", data[key]["filename"])
    for j in range(len(data[key]["regions"])):
        x_lst = np.asarray(data[key]["regions"][j]["shape_attributes"]["all_points_x"])
        y_lst = np.asarray(data[key]["regions"][j]["shape_attributes"]["all_points_y"])
        pts = np.asarray([[int(x), int(y)] for x,y in zip(x_lst, y_lst)])
        
        mask = createMask(frame, pts)            
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        x = 0
        y = 0
        w =0
        h =0
        for i in range(len(contours)):
            x,y,w,h = cv2.boundingRect(contours[i])
        manual_data ={
        "roi" : [x,y,x+w,y+h],
        "center" : [x +w/2,y+h/2],
        "pts" : pts}    
        manual_data_array.append(manual_data)


    ie = IECore()
    net = IENetwork(model='/Users/vedanshu/Downloads/customrcnnident.xml', weights='/Users/vedanshu/Downloads/customrcnnident.bin')
    supported_layers = ie.query_network(net, "CPU")
    not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    if len(not_supported_layers) != 0:
        log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                  format(args.device, ', '.join(not_supported_layers)))
        log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                  "or --cpu_extension command line argument")
        sys.exit(1)

    img_info_input_blob = None
    feed_dict = {}
    for blob_name in net.inputs:
        if len(net.inputs[blob_name].shape) == 4:
            input_blob = blob_name
        elif len(net.inputs[blob_name].shape) == 2:
            img_info_input_blob = blob_name
        else:
            raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported"
                               .format(len(net.inputs[blob_name].shape), blob_name))

    exec_net = ie.load_network(network=net, device_name="CPU")
    n, c, h, w = net.inputs[input_blob].shape
    if img_info_input_blob:
            feed_dict[img_info_input_blob] = [h, w, 1]

    cur_request_id = 0


    in_frame = cv2.resize(frame, (w, h))
    in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    in_frame = in_frame.reshape((n, c, h, w))
    feed_dict[input_blob] = in_frame
    exec_net.start_async(request_id=cur_request_id, inputs=feed_dict)

    while True:

        if exec_net.requests[cur_request_id].wait(-1) == 0:
            res_box = exec_net.requests[cur_request_id].outputs['detection_output']
            res_mask = exec_net.requests[cur_request_id].outputs['masks']
            for index in range(res_box[0][0].shape[0]):
                # Draw only objects when probability more than specified threshold
                obj = res_box[0][0][index]
                mask = res_mask[index][0]
                if obj[2] > 0.5:
                    xmin = int(obj[3] * frame_w)
                    ymin = int(obj[4] * frame_h)
                    xmax = int(obj[5] * frame_w)
                    ymax = int(obj[6] * frame_h)
                    boxW = xmax - xmin
                    boxH = ymax - ymin
                    # embed()
                    mask = cv2.resize(mask, (boxW, boxH))
                    mask = (mask > 0.5)
                    roi = frame[ymin:ymax, xmin:xmax][mask]
                    blended = (0.6 * roi).astype("uint8")
                    frame[ymin:ymax, xmin:xmax][mask] = blended*[0,0,0.8]

                    polyA = mask_to_polygon(np.array(mask,dtype=np.uint8))

                    for i in range(len(polyA)):
                        polyA[i][0] = polyA[i][0] + xmin
                        polyA[i][1] = polyA[i][1] + ymin

                    for j in range(len(manual_data_array)):
                        polyB =  manual_data_array[j]["pts"]
                        iou = calculate_iou(polyA, polyB)
                        if iou > 0.3:
                            print(j, ": ", iou)
                            cv2.putText(frame, str(round(iou*100,2)) + " %",(int((xmin+xmax)/2),int((ymin+ymax)/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) , 2, cv2.LINE_AA)
                            cv2.putText(frame, str(j),(int((xmin+xmax)/2),int((ymin+ymax)/2)+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) , 2, cv2.LINE_AA) 
                            polyA = np.array(polyA)
                            polyB = np.array(polyB)
                            cv2.polylines(frame,np.int32([polyA]),True,(0,255,0), thickness=2)
                            cv2.polylines(frame,np.int32([polyB]),True,(255,0,0), thickness=2)
                        
                    # class_id = int(obj[1])
                    # # Draw box and label\class_id
                    # color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
                    # cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    # cv2.putText(frame, str(class_id) + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
                    #             cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

            cv2.imwrite("data/out/"+str(i)+".jpg", frame)
            break
stop = True
t1.join()
print("Processing done...")

with open('load.csv', 'w') as f:
    for item in utilzation:
        f.write(str(item)+'\n')



