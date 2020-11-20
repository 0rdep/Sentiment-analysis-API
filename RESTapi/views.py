import glove
import os
import requests
import shutil
from django.shortcuts import render
from rest_framework.views import APIView
from django.http import JsonResponse
from rest_framework.response import Response

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from rest_framework.permissions import AllowAny
from django.views.generic.base import TemplateView

from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema
from .serializers import (
    ClassificationGetResponseSerializer,
    ClassificationPostRequestSerializer,
    ClassificationPostSuccessResponseSerializer,
    ClassificationPostErrorResponseSerializer
)

# category labels
main_labels = ['confident', 'unconfident',
               'pos_hp', 'neg_hp',
               'interested', 'uninterested',
               'happy', 'unhappy',
               'friendly', 'unfriendly'
               ]

# to dictionary
label_dict = dict(zip(main_labels, range(0, len(main_labels))))

# inverting label_dict
inv_label = {v: k for k, v in label_dict.items()}

# Load the pretrained model
model = load_model('RESTapi/tfmodel.h5')
#model = load_model('C:\\Users\\Acer\\Documents\\coding\\Upwork\\Sent Rest-API\\RESTapi\\tfmodel.h5')

# GloVe
glove_dir = '/app/data/RNN/'
glove_100k_50d = 'glove.first-100k.6B.50d.txt'
glove_100k_50d_path = os.path.join(glove_dir, glove_100k_50d)

# These are temporary files if we need to download it from the original source (slow)
data_cache = '/app/data/cache'
glove_full_tar = 'glove.6B.zip'
glove_full_50d = 'glove.6B.50d.txt'

# force_download_from_original=False
download_url = 'http://redcatlabs.com/downloads/deep-learning-workshop/notebooks/data/RNN/'+glove_100k_50d
original_url = 'http://nlp.stanford.edu/data/'+glove_full_tar

if not os.path.isfile(glove_100k_50d_path):
    if not os.path.exists(glove_dir):
        os.makedirs(glove_dir)

    # First, try to download a pre-prepared file directly...
    response = requests.get(download_url, stream=True)
    if response.status_code == requests.codes.ok:
        print("Downloading 42Mb pre-prepared GloVE file from RedCatLabs")
        with open(glove_100k_50d_path, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
    else:
        # But, for some reason, RedCatLabs didn't give us the file directly
        if not os.path.exists(data_cache):
            os.makedirs(data_cache)

        if not os.path.isfile(os.path.join(data_cache, glove_full_50d)):
            zipfilepath = os.path.join(data_cache, glove_full_tar)
            if not os.path.isfile(zipfilepath):
                print("Downloading 860Mb GloVE file from Stanford")
                response = requests.get(download_url, stream=True)
                with open(zipfilepath, 'wb') as out_file:
                    shutil.copyfileobj(response.raw, out_file)
            if os.path.isfile(zipfilepath):
                print("Unpacking 50d GloVE file from zip")
                import zipfile
                zipfile.ZipFile(zipfilepath, 'r').extract(
                    glove_full_50d, data_cache)

        with open(os.path.join(data_cache, glove_full_50d), 'rt') as in_file:
            with open(glove_100k_50d_path, 'wt') as out_file:
                print("Reducing 50d GloVE file to first 100k words")
                for i, l in enumerate(in_file.readlines()):
                    if i >= 100000:
                        break
                    out_file.write(l)

        # Get rid of tarfile source (the required text file itself will remain)
        # os.unlink(zipfilepath)
        #os.unlink(os.path.join(data_cache, glove_full_50d))

# Due to size constraints, only use the first 100k vectors (i.e. 100k most frequently used words)
word_embedding = glove.Glove.load_stanford(glove_100k_50d_path)
word_embedding.word_vectors.shape


def get_embedding_vec(word):
    """
    return : embedding vector of a word
    """
    idx = word_embedding.dictionary.get(word.lower(), -1)

    if idx < 0:
        return np.zeros((50, ), dtype='float32')  # UNK EMBEDDING_DIM=50

    return word_embedding.word_vectors[idx]


def predict_class(word, valence, arousal, dominance, model):
    """
    a function that predicts sentiments from words

    ARGUMENTS : 

    word : our word (example: happy, tolerant ...)

    valence,arousal,dominance,quadrant : important features that boost the accuracy of the model

    return : class predicted

    """

    # creating the embedding of the word
    embedding_features = get_embedding_vec(word)

    feats_without_embedding = [valence, arousal, dominance]

    # creating our X that contains all features (word_embeddings, VADQ)
    x_t = np.concatenate((feats_without_embedding, embedding_features)).reshape(
        1, 53)  # 54 if we are using quadrant as parameter

    # predictions (we will take saved models of each fold and do an average of predictions)
    prediction = model.predict(x=x_t, verbose=1)

    return inv_label[np.argmax(prediction)]


classificationPostSuccessResponseSerializer = openapi.Response(
    'Success classification', ClassificationPostSuccessResponseSerializer)

classificationPostErrorResponseSerializer = openapi.Response(
    'Error on classification', ClassificationPostErrorResponseSerializer)

classificationGetResponse = openapi.Response(
    'Post request information', ClassificationGetResponseSerializer)

# Classification call class
class Classification(APIView):

    permission_classes = (AllowAny,)

    @swagger_auto_schema(
        operation_description="Get classification request information", 
        responses={
            200: classificationGetResponse
        }
    )
    def get(self, request, format=None):
        """API View"""
        data_view = ['word', 'valence', 'arousal', 'dominance']
        return Response({'url': '/classification',
                         'type': 'POST',
                         'data': data_view,
                         'code': 200,
                         'status': 'Success'
                         })

    @swagger_auto_schema(
        request_body=ClassificationPostRequestSerializer,
        responses={
            200: classificationPostSuccessResponseSerializer,
            400: classificationPostErrorResponseSerializer
        }
    )
    def post(self, request):
        try:
            # word to predict with VAD values
            word = request.POST.get('word')
            valance = request.POST.get('valance')
            arousal = request.POST.get('arousal')
            dominance = request.POST.get('dominance')

            # calling prediction function
            pred = predict_class(word, valance, arousal, dominance, model)
            print('Prediction: {}'.format(pred))

            return JsonResponse({'message': 'Prediction: ' + str(pred), 'code': 200, 'status': 'Success'})
        except Exception as e:
            print('email error:', e)
            return JsonResponse({'message': 'Something went wrong', 'code': 400, 'status': 'Error', 'error': str(e)})


class Home(TemplateView):
    template_name = ('home.html')
