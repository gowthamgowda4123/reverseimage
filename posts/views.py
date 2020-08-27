import json

from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
# from django.http import HttpResponse


import os
import numpy as np
from PIL import Image
from .feature_extractor import FeatureExtractor
import glob
import pickle
from datetime import datetime
from django.shortcuts import render
from .converter import *

# app = Flask(__name__)

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in glob.glob("static/feature/*"):
    features.append(pickle.load(open(feature_path, 'rb')))
    img_paths.append('static/img/' + os.path.splitext(os.path.basename(feature_path))[0] + '.jpg')


# @app.route('/', methods=['GET', 'POST'])
#@api_view(['GET', 'PUT'])
def index(request):
    if request.method == 'POST':
        print("hii")
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        content = body['content']
        print(content)

        file = request.files['query_img']

        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat() + "_" + file.filename
        img.save(uploaded_img_path)

        query = fe.extract(img)
        dists = np.linalg.norm(features - query, axis=1)  # Do search
        ids = np.argsort(dists)[:30]  # Top 30 results
        scores = [(dists[id], img_paths[id]) for id in ids]

        return render(request, 'index.html',
                      query_path=uploaded_img_path,
                      scores=scores)
    else:
        print("bye")
        return render('index.html')





# if __name__=="__main__":
#     app.run("0.0.0.0")
#
#
# def index(request):
#     return render(request, 'posts/index.html')
