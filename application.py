# import os
# import PIL
# import numpy


# from numpy.lib.function_base import average


# from numpy import zeros
# from numpy import asarray

# from mrcnn.config import Config

# from mrcnn.model import MaskRCNN

# from skimage.draw import polygon2mask
# from skimage.io import imread

# from datetime import datetime


# from io import BytesIO
# from mrcnn.utils import extract_bboxes
# from numpy import expand_dims
# from matplotlib import pyplot
# from matplotlib.patches import Rectangle
# from keras.backend import clear_session
# import json
# from flask import Flask, flash, request,jsonify, redirect, url_for
# from werkzeug.utils import secure_filename

# from skimage.io import imread
# from mrcnn.model import mold_image

# import tensorflow as tf
# import sys

# from PIL import Image


# global _model
# global _graph
# global cfg
# ROOT_DIR = os.path.abspath("./")
# WEIGHTS_FOLDER = "./weights"

# from flask_cors import CORS, cross_origin

# sys.path.append(ROOT_DIR)

# MODEL_NAME = "mask_rcnn_hq"
# WEIGHTS_FILE_NAME = 'maskrcnn_15_epochs.h5'

# application=Flask(__name__)
# cors = CORS(application, resources={r"/*": {"origins": "*"}})


# class PredictionConfig(Config):
# 	# define the name of the configuration
# 	NAME = "floorPlan_cfg"
# 	# number of classes (background + door + wall + window)
# 	NUM_CLASSES = 1 + 3
# 	# simplify GPU config
# 	GPU_COUNT = 1
# 	IMAGES_PER_GPU = 1

# @application.before_first_request
# def load_model():
# 	global cfg
# 	global _model
# 	model_folder_path = os.path.abspath("./") + "/mrcnn"
# 	weights_path= os.path.join(WEIGHTS_FOLDER, WEIGHTS_FILE_NAME)
# 	cfg=PredictionConfig()
# 	print(cfg.IMAGE_RESIZE_MODE)
# 	print('==============before loading model=========')
# 	_model = MaskRCNN(mode='inference', model_dir=model_folder_path,config=cfg)
# 	print('=================after loading model==============')
# 	_model.load_weights(weights_path, by_name=True)
# 	global _graph
# 	_graph = tf.get_default_graph()


# def myImageLoader(imageInput):
# 	image =  numpy.asarray(imageInput)


# 	h,w,c=image.shape
# 	if image.ndim != 3:
# 		image = skimage.color.gray2rgb(image)
# 		if image.shape[-1] == 4:
# 			image = image[..., :3]
# 	return image,w,h

# def getClassNames(classIds):
# 	result=list()
# 	for classid in classIds:
# 		data={}
# 		if classid==1:
# 			data['name']='wall'
# 		if classid==2:
# 			data['name']='window'
# 		if classid==3:
# 			data['name']='door'
# 		result.append(data)

# 	return result
# def normalizePoints(bbx,classNames):
# 	normalizingX=1
# 	normalizingY=1
# 	result=list()
# 	doorCount=0
# 	index=-1
# 	doorDifference=0
# 	for bb in bbx:
# 		index=index+1
# 		if(classNames[index]==3):
# 			doorCount=doorCount+1
# 			if(abs(bb[3]-bb[1])>abs(bb[2]-bb[0])):
# 				doorDifference=doorDifference+abs(bb[3]-bb[1])
# 			else:
# 				doorDifference=doorDifference+abs(bb[2]-bb[0])


# 		result.append([bb[0]*normalizingY,bb[1]*normalizingX,bb[2]*normalizingY,bb[3]*normalizingX])
# 	return result,(doorDifference/doorCount)


# def turnSubArraysToJson(objectsArr):
# 	result=list()
# 	for obj in objectsArr:
# 		data={}
# 		data['x1']=obj[1]
# 		data['y1']=obj[0]
# 		data['x2']=obj[3]
# 		data['y2']=obj[2]
# 		result.append(data)
# 	return result


# @application.route('/',methods=['GET', 'POST'])
# def prediction():

# 	if request.method == "GET":
# 		return """
# 			<h1>Mask R-CNN Floorplan Predictor</h1>
# 			<p>Send a <code>POST</code> to this URL with a form-file named <code>image</code>.</p>
# 			<form method="post" enctype="multipart/form-data">
# 				<input type="file" name="image" accept="image/*">
# 				<button type="submit">Upload + Predict</button>
# 			</form>
# 		""", 200

# 	global cfg
# 	imagefile = PIL.Image.open(request.files['image'].stream)
# 	image,w,h=myImageLoader(imagefile)
# 	print(h,w)
# 	scaled_image = mold_image(image, cfg)
# 	sample = expand_dims(scaled_image, 0)

# 	global _model
# 	global _graph
# 	with _graph.as_default():
# 		r = _model.detect(sample, verbose=0)[0]

# 	#output_data = model_api(imagefile)

# 	data={}
# 	bbx=r['rois'].tolist()
# 	temp,averageDoor=normalizePoints(bbx,r['class_ids'])
# 	temp=turnSubArraysToJson(temp)
# 	data['points']=temp
# 	data['classes']=getClassNames(r['class_ids'])
# 	data['Width']=w
# 	data['Height']=h
# 	data['averageDoor']=averageDoor
# 	return jsonify(data)


# if __name__ =='__main__':
# 	application.debug=True
# 	print('===========before running==========')
# 	application.run()
# 	print('===========after running==========')


import os
import sys
import numpy as np
from io import BytesIO
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
import json

from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import extract_bboxes
from mrcnn.model import mold_image
from numpy import expand_dims

# ============================
# Configuration
# ============================
WEIGHTS_FOLDER = "./weights"
WEIGHTS_FILE_NAME = 'maskrcnn_15_epochs.h5'


class PredictionConfig(Config):
    NAME = "floorPlan_cfg"
    NUM_CLASSES = 1 + 3  # background + wall, window, door
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# ============================
# Flask App Setup
# ============================
application = Flask(__name__)
CORS(application, resources={r"/*": {"origins": "*"}})

global_model = None
global_graph = None
config = None


@application.before_first_request
def load_model():
    global global_model, global_graph, config
    config = PredictionConfig()
    model_dir = os.path.abspath("./mrcnn_model")
    weights_path = os.path.join(WEIGHTS_FOLDER, WEIGHTS_FILE_NAME)
    print("Loading Mask R-CNN model from:", weights_path)
    global_model = MaskRCNN(mode='inference', model_dir=model_dir, config=config)
    global_model.load_weights(weights_path, by_name=True)
    global_graph = tf.get_default_graph()
    print("Model loaded successfully.")


# ============================
# Helper Functions
# ============================

def myImageLoader(image_input):
    image = np.asarray(image_input)
    if image.ndim != 3:
        # convert grayscale to RGB
        from skimage import color
        image = color.gray2rgb(image)
    if image.shape[-1] == 4:
        # drop alpha channel
        image = image[..., :3]
    h, w = image.shape[:2]
    return image, w, h


def getClassNames(class_ids):
    mapping = {1: 'wall', 2: 'window', 3: 'door'}
    return [{'name': mapping.get(cid, 'unknown')} for cid in class_ids]


def normalizePoints(bboxes, class_ids):
    result = []
    door_diff_sum = 0
    door_count = 0
    for idx, bb in enumerate(bboxes):
        cid = class_ids[idx]
        x1, y1, x2, y2 = bb
        result.append([y1, x1, y2, x2])
        # accumulate door size for average
        if cid == 3:
            door_count += 1
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            door_diff_sum += max(width, height)
    avg_door = (door_diff_sum / door_count) if door_count else 0
    return result, avg_door


def turnSubArraysToJson(arrays):
    return [{'x1': a[1], 'y1': a[0], 'x2': a[3], 'y2': a[2]} for a in arrays]


# ============================
# Routes
# ============================
@application.route('/', methods=['GET', 'POST'])
def prediction():
    if request.method == 'GET':
        # simple upload form
        return (
            """
            <h1>Mask R-CNN Floorplan Predictor</h1>
            <p>Upload an image to get wall, window, and door detections.</p>
            <form method="post" enctype="multipart/form-data">
              <input type="file" name="image" accept="image/*" required>
              <button type="submit">Upload + Predict</button>
            </form>
            """, 200
        )

    # POST: perform prediction
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    img_stream = request.files['image'].stream
    pil_img = Image.open(img_stream).convert('RGB')
    image, w, h = myImageLoader(pil_img)
    molded = mold_image(image, config)
    sample = expand_dims(molded, 0)

    with global_graph.as_default():
        r = global_model.detect(sample, verbose=0)[0]

    bboxes = r['rois'].tolist()
    norm_boxes, avg_door = normalizePoints(bboxes, r['class_ids'])
    json_boxes = turnSubArraysToJson(norm_boxes)

    response = {
        'points': json_boxes,
        'classes': getClassNames(r['class_ids']),
        'Width': w,
        'Height': h,
        'averageDoor': avg_door
    }

    # Save to file
    with open('disney_1.json', 'w') as f:
        json.dump(response, f, indent=4)

    return jsonify(response)


# ============================
# Entry Point
# ============================
if __name__ == '__main__':
    application.debug = True
    application.run(host='0.0.0.0', port=5000)