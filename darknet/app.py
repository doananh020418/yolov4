from base64 import b64decode, b64encode
import cv2
import numpy as np
import PIL
import io
import time
import matplotlib.pyplot as plt
import glob
import pandas as pd
import os
import glob
import cv2
import threading
import urllib.request
from PIL import Image
from flask import Flask, Response, request, jsonify, render_template
from twilio.rest import Client
from darknet import *
app = Flask(__name__)
import json

# load in our YOLOv4 architecture network
network, class_names, class_colors = load_network("cfg/yolov4-custom.cfg", "data/obj.data", "81.weights")
width = network_width(network)
height = network_height(network)

account_sid = os.environ['ACfbd08622bcdb0c17a13c27ae0f11ea84']
auth_token = os.environ['7d641a83b143605c7d9eeb136b74f9c9']
client = Client(account_sid, auth_token)



# darknet helper function to run detection on image
def darknet_helper(img, width, height):
  darknet_image = make_image(width, height, 3)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_resized = cv2.resize(img_rgb, (width, height),
                              interpolation=cv2.INTER_LINEAR)

  # get image ratios to convert bounding boxes to proper size
  img_height, img_width, _ = img.shape
  width_ratio = img_width/width
  height_ratio = img_height/height

  # run model on darknet style image to get detections
  copy_image_from_bytes(darknet_image, img_resized.tobytes())
  detections = detect_image(network, class_names, darknet_image)
  free_image(darknet_image)
  return detections, width_ratio, height_ratio


def bbox_to_bytes(bbox_array):
  """
  Params:
          bbox_array: Numpy array (pixels) containing rectangle to overlay on video stream.
  Returns:
        bytes: Base64 image byte string
  """
  # convert array into PIL image
  bbox_PIL = PIL.Image.fromarray(bbox_array, 'RGBA')
  iobuf = io.BytesIO()
  # format bbox into png for return
  bbox_PIL.save(iobuf, format='png')
  # format return string
  bbox_bytes = 'data:image/png;base64,{}'.format((str(b64encode(iobuf.getvalue()), 'utf-8')))

  return bbox_bytes  

class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
    def run(self):
        print("Starting " + self.previewName)
        camPreview(self.previewName, self.camID)

def camPreview(previewName, camID):
    cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,300)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,300)
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        rval = False
        print('False')
    rval, frame = cam.read()
    while rval:
        rval, frame = cam.read()
        detections, width_ratio, height_ratio = darknet_helper(frame, width, height)
        if len(detections) > 0:
          tmp = []
          for _,conf,_ in detections:
            tmp.append(float(conf))
          if max(tmp) > 50:
            message = client.messages \
                .create(
                     body="Fire warning!!!",
                     from_='+18508314922',
                     to='+84946318159'
                 )
          
          
        else:
          pass
        

# @app.route('/')
# def home():
#     ret = 'hello'
#     return ret
# sc = []
# @app.route('/send_rtsp', methods=['GET', 'POST'])
# def send_rtsp():
#     link = request.args.get('rtsp_link')
#     #print(link)
#     thread = camThread('cam',link)
#     thread.start()

#     return 'Done'

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5002, debug=False)
import time

frame = cv2.imread(r"D:\fire smoke dataset\Fire-Smoke-Test\Fire\1_20.jpg")
detections, width_ratio, height_ratio = darknet_helper(frame, width, height)
print(detections[0])

