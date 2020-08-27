import json

from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
# from django.http import HttpResponse


import os
import numpy as np
from PIL import Image
from tensorflow.python.client import session

from .feature_extractor import FeatureExtractor
import glob
import pickle
from datetime import datetime
from django.shortcuts import render
from .converter import *
from django.conf.urls import url
import os
import requests
import tensorflow as tf

from django.test import Client
from django.template import loader
client = Client()
response = client.post(url, content_type='application/json')

# app = Flask(__name__)

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in glob.glob("static/feature/*"):
    features.append(pickle.load(open(feature_path, 'rb')))
    img_paths.append('static/img/' + os.path.splitext(os.path.basename(feature_path))[0] + '.jpg')

# @app.route('/', methods=['GET', 'POST'])
# @api_view(['GET', 'PUT'])


global graph
graph = tf.get_default_graph()


def index(request):
    if request.method == 'POST':
        print("hii")

        # body_unicode = request.POST#.decode('utf-8') # changed request.body to request.post
        # body = json.loads(body_unicode)
        # content = body['content']
        # print(content)

        file = request.FILES['query_img']

        img = Image.open(request.FILES['query_img'])  # PIL image #file.stream

        # form = ImageForm(request.POST, request.FILES)
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat() + "_" + str(request.FILES['query_img'])
        img.save(uploaded_img_path)
        with graph.as_default():
            query = fe.extract(img)
            dists = np.linalg.norm(features - query, axis=1)  # Do search
            ids = np.argsort(dists)[:30]  # Top 30 results
            scores = [(dists[id], img_paths[id]) for id in ids]
            query_path = uploaded_img_path
            scores = scores

        context = {
            'query_path': query_path,
            'scores':scores

        }

        return render(request, 'index.html',
                      context)
    else:
        print("bye")
        return render(request, 'index.html')

# if __name__=="__main__":
#     app.run("0.0.0.0")
#
#
# def index(request):
#     return render(request, 'posts/index.html')
