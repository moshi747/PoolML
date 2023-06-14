import cv2
import numpy as np
import math
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import copy

GREEN = np.array([64,128,0], dtype = np.uint8)
BLUE = np.array([255,128,32], dtype = np.uint8)
GRAY = np.array([206,186,186], dtype = np.uint8)

model = YOLO('Vision/best.pt')

        
def get_homography(img_orig, c):
    
    if c == 'GREEN':
        color=GREEN
    elif c == 'BLUE':
        color=BLUE
    else:
        color=GRAY
    
    img = copy.copy(img_orig)
    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low, high = get_limits(color, error = 20)

    if color is GRAY:
        mask = cv2.inRange(img, color-30, color+30)
    else:
        mask = cv2.inRange(hsvImage, low, high)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    for i in range(len(contours)):
        if cv2.contourArea(contours[i])>max_area:
            max_ind = i
            max_area = cv2.contourArea(contours[i])

    hull = cv2.convexHull(contours[max_ind])

    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    cv2.drawContours(drawing, [hull], 0, (0,128,0),1)

    drawing = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)

    lines = cv2.HoughLines(drawing, 1, np.pi / 180, 120, None, 0, 0)

    drawing2 = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    linesavg = []

    for line in lines:
        temp = True
        for lineavg in linesavg:
            if abs((line[0][0]-lineavg[0][0])/img.shape[1])<0.1 and abs(line[0][1]-lineavg[0][1])<0.2:
                lineavg[0]=(line[0]+lineavg[0])/2
                temp = False
                break
        if temp:
            linesavg.append(line)

    lines = []

    if lines is not None:
        for i in range(len(linesavg)):
            rho = linesavg[i][0][0]
            theta = linesavg[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            if abs(a)<abs(b):
                d = max(abs((x0-img.shape[1])/b), abs(x0/b))
            else:
                d = max(abs((y0-img.shape[0])/a), abs(y0/a))
            pt1 = (int(x0 + d*(-b)), int(y0 + d*(a)))
            pt2 = (int(x0 - d*(-b)), int(y0 - d*(a)))
            lines.append((pt1, pt2))
            cv2.line(img, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)

    tc = []

    for j in range(1, 4):
        x1, y1 = lines[0][0]
        x2, y2 = lines[0][1]
        w1, z1 = lines[j][0]
        w2, z2 = lines[j][1]
        N = (z1-y1)*(x2-x1)-(w1-x1)*(y2-y1)
        D = (w2-w1)*(y2-y1)-(z2-z1)*(x2-x1)
        if abs(N) < abs(D) and N*D > 0:
            s = N/D
            pt = np.array([int(w1+s*(w2-w1)), int(z1+s*(z2-z1))])
            if pt[0]>0 and pt[0]<img.shape[1] and pt[1]>0 and pt[1]<img.shape[0]:
                tc.append(pt)
        else:
            i = j

    for j in range(1, 4):
        if i != 4-j:
            x1, y1 = lines[i][0]
            x2, y2 = lines[i][1]
            w1, z1 = lines[4-j][0]
            w2, z2 = lines[4-j][1]
            N = (z1-y1)*(x2-x1)-(w1-x1)*(y2-y1)
            D = (w2-w1)*(y2-y1)-(z2-z1)*(x2-x1)
            if abs(N) < abs(D) and N*D > 0:
                s = N/D
                pt = np.array([int(w1+s*(w2-w1)), int(z1+s*(z2-z1))])
                if pt[0]>0 and pt[0]<img.shape[1] and pt[1]>0 and pt[1]<img.shape[0]:
                    tc.append(pt)

    tc = order_corners(tc)
    tc = np.array(tc)

    
    table = np.zeros((320, 640, 3), dtype=np.uint8)
    pockets = np.array([[0,0], [0,640], [320,640], [320,0]])

    h, status = cv2.findHomography(tc, pockets)
    return h
    
    
def get_limits(color, error = 10):
    hsv_c = cv2.cvtColor(np.array([[color]]), cv2.COLOR_BGR2HSV)
    low = [hsv_c[0][0][0] - error, 80, 80]
    high = [hsv_c[0][0][0] + error, 255, 255]
    
    low[0] = max(low[0], 0)
    high[0] = min(high[0], 255)
    
    low = np.array(low, dtype = np.uint8)
    high = np.array(high, dtype = np.uint8)
     
    return low, high

def order_corners(corners):
    shortest_dist = np.linalg.norm(corners[0]-corners[1])
    point = 0
    
    for i in range(1,4):
        if np.linalg.norm(corners[i]-corners[(i+1)%4]) < shortest_dist:
            shortest_dist=np.linalg.norm(corners[i]-corners[(i+1)%4])
            point = i
    
    if corners[point][1] < corners[(point+1)%4][1]:
        for i in range(4):
            corners[i] = corners[(point+1+i)%4]
    else:
        for i in range(4):
            corners[i] = corners[(point-i)%4]
    return corners

def apply_holo(h, point):
    new = np.hstack([np.array(point), np.array([1])])
    ans = np.matmul(h, new)
    ans = ans/ans[2]
    return ans[:2]

def export_yolo_data(
    samples,
    export_dir,
    classes,
    label_field = "ground_truth",
    split = None
    ):

    if type(split) == list:
        splits = split
        for split in splits:
            export_yolo_data(
                samples,
                export_dir,
                classes,
                label_field,
                split
            )
    else:
        if split is None:
            split_view = samples
            split = "val"
        else:
            split_view = samples.match_tags(split)

        split_view.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=label_field,
            classes=classes,
            split=split
        )
        
def get_coords(img_orig, h):
    
    results = model.predict(source=img_orig, save=True, save_txt=True)

    centers = []
    for box in results[0].boxes.data:
        x = (box[0]+box[2])/2
        y = (box[1]+box[3])/2
        x, y = int(x.item()), int(y.item())
        centers.append((x,y))
    
    new_centers = []

    for center in centers:
        new_center = apply_holo(h, center)
        if new_center[0]<0 or new_center[0]>320 or new_center[1]<0 or new_center[1]>640:
            continue
        new_center = new_center.astype('int')
        new_center = tuple(new_center)
        new_centers.append(new_center)
    
    return (new_centers, centers)
      

def reorder(centers, order):

    new_centers = []
    for i in order:
        new_centers.append(centers[i])

    return new_centers
