#/usr/bin/env python

import numpy as np
import math
import cv2
import cv2.cv as cv
from video import create_capture
from common import clock, draw_str

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
  
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        sub_x1 = int(x1 + math.ceil(0.25*(x2-x1)))
        sub_x2 = int(x2 - math.ceil(0.25*(x2-x1)))
        sub1_y1 = y1
        sub1_y2 = int(y1 + math.ceil(.2*(y2-y1)))
        sub2_y1 = int(y1 + math.ceil(.55*(y2-y1)))
        sub2_y2 = int(y2 - math.ceil(0.1*(y2-y1)))
        cv2.rectangle(img, (sub_x1, sub1_y1), (sub_x2, sub1_y2), color, 2)
        cv2.rectangle(img, (sub_x1, sub2_y1), (sub_x2, sub2_y2), color, 2)
        return [sub_x1,sub_x2,sub1_y1,sub2_y2,sub2_y1,sub2_y2]

def issmallest(rects,smallest_bb):
    small_lenx = smallest_bb[0][2] - smallest_bb[0][0]
    small_leny = smallest_bb[0][3] - smallest_bb[0][1]
    for x1 , y1, x2, y2 in rects:
        lx=x2-x1
        ly=y2-y1
        print(lx,ly)
        print(small_lenx,small_leny)
        if lx < small_lenx :
            if ly < small_leny:
                smallest_bb = np.array([[x1,y1,x2,y2]])
    return smallest_bb

if __name__ == '__main__':
    import sys, getopt

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try: video_src = video_src[0]
    except: video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "haarcascade_frontalface_alt2.xml")

    cascade = cv2.CascadeClassifier(cascade_fn)

    cam = create_capture(video_src, fallback='synth:bg=../cpp/lena.jpg:noise=0.05')
    smallest_bb = np.array([[0, 0, 400, 400]])
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        t = clock()
        rects = detect(gray, cascade)
        print(rects)
        smallest_bb = issmallest(rects, smallest_bb)
        print(smallest_bb)
        vis = img.copy()
        draw_rects(vis, smallest_bb, (0, 255, 0))
        dt = clock() - t

        #draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv2.imshow('facedetect', vis)

        if 0xFF & cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()

