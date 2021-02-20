import iewrap

import time

import cv2
import numpy as np
import image_compose as img_cmp
import pyfakewebcam

imgBuf = {}
camera = pyfakewebcam.FakeWebcam('/dev/video2', 640, 480)

def fnc_mosaic(src, ratio=0.1):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

def fnc_area_mosaic(src, top_left, bottom_right, ratio=0.1):
    dst = src.copy()
    x = top_left[0]
    y = top_left[1]
    height = bottom_right[1] - top_left[1]
    width = bottom_right[0] - top_left[0]
    dst[y:y + height, x:x + width] = fnc_mosaic(dst[y:y + height, x:x + width], ratio)
    return dst

def callback(infId, output):
    global imgBuf

    output = output.reshape((200,7))
    img = imgBuf.pop(infId)
    img_h, img_w, _ = img.shape

    img_mosaic = fnc_area_mosaic(img, (0, 0), (img_w, img_h),0.025)

    for obj in output:
        imgid, clsid, confidence, x1, y1, x2, y2 = obj
        if confidence>0.3:              # Draw a bounding box and label when confidence>0.8
            x1 = int(x1 * img_w)
            y1 = int(y1 * img_h)
            x2 = int(x2 * img_w)
            y2 = int(y2 * img_h)

            detected_face = img[y1:y2,x1:x2]
            img_mosaic[y1:y2,x1:x2] = detected_face

    #cv2.imshow('result',img_mosaic)
    #cv2.imshow('result',im_back)
    #cv2.waitKey(1)

    img_mosaic = cv2.cvtColor(img_mosaic, cv2.COLOR_BGR2RGB)
    camera.schedule_frame(img_mosaic)
    time.sleep(0.033)

def main():
    global imgBuf

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH , 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    #ie = iewrap.ieWrapper(r'/home/pi/openvino/person_detect_demo/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml', 'MYRIAD', 10)
    ie = iewrap.ieWrapper(r'/home/pi/openvino/models/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml', 'MYRIAD', 10)
    ie.setCallback(callback)

    while True:
        ret, img = cap.read()
        if ret==False:
            break

        refId = ie.asyncInfer(img)     # Inference
        imgBuf[refId]=img

if __name__ == '__main__':
    main()
