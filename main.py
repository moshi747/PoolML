from flask import Flask, request, jsonify
from PIL import Image
import Vision.vision as vision
from pool import PoolAI, balls_to_obs, rescale, view_shots
import cv2
import matplotlib.pyplot as plt
import copy
import numpy
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return '''Server Works!<hr>
<form action="/processing" method="POST" enctype="multipart/form-data">
<input type="file" name="image">
<button>OK</button>
</form>
'''

@app.route('/processing', methods=['POST'])
def process():
    file = request.files['image']
    
    img = Image.open(file.stream)
    img_ar = numpy.asarray(img)
    img_ar = img_ar[:, :, ::-1]
    
    h = vision.get_homography(img_ar, 'BLUE')
    t_coords, o_coords = vision.get_coords(img_ar, h)
    
    # data = file.stream.read()
    # data = base64.encodebytes(data)
    # data = base64.b64encode(data).decode()   
    
    return str(t_coords)
    
if __name__ == '__main__':
    app.run(debug=True)