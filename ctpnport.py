
import sys
import numpy as np
class cfg:
    MEAN=np.float32([102.9801, 115.9465, 122.7717])
    TEST_GPU_ID=0
    SCALE=600
    MAX_SCALE=1000

    LINE_MIN_SCORE=0.7
    TEXT_PROPOSALS_MIN_SCORE=0.7
    TEXT_PROPOSALS_NMS_THRESH=0.3
    MAX_HORIZONTAL_GAP=50
    TEXT_LINE_NMS_THRESH=0.3
    MIN_NUM_PROPOSALS=2
    MIN_RATIO=1.2
    MIN_V_OVERLAPS=0.7
    MIN_SIZE_SIM=0.7
    TEXT_PROPOSALS_WIDTH=16

def init():
    sys.path.insert(0, "./CTPN/tools")
    sys.path.insert(0, "./CTPN/caffe/python")
    sys.path.insert(0, "./CTPN/src")
init()

from other import draw_boxes, resize_im, CaffeModel
import cv2, os, caffe
from detectors import TextProposalDetector, TextDetector
import os.path as osp
from utils.timer import Timer



def ctpnSource():
    DEMO_IMAGE_DIR = "img/"
    NET_DEF_FILE = "CTPN/models/deploy.prototxt"
    MODEL_FILE = "CTPN/models/ctpn_trained_model.caffemodel"
    caffe.set_mode_gpu()
    caffe.set_device(cfg.TEST_GPU_ID)
    # initialize the detectors
    text_proposals_detector = TextProposalDetector(CaffeModel(NET_DEF_FILE, MODEL_FILE))
    text_detector = TextDetector(text_proposals_detector)
    return text_detector

def getCharBlock(text_detector,im):
    im, f=resize_im(im, cfg.SCALE, cfg.MAX_SCALE)
    cv2.imshow("src", im)
    tmp = im.copy()
    #timer=Timer()
    #timer.tic()
    text_lines=text_detector.detect(im)

    #print "Number of the detected text lines: %s"%len(text_lines)
    #print "Time: %f"%timer.toc()

    text_recs = draw_boxes(tmp, text_lines, caption='im_name', wait=True)
    return tmp,text_recs


    



