from django.shortcuts import render
from rest_framework.views import APIView
from django.http import JsonResponse

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from rest_framework.permissions import AllowAny
from django.views.generic.base import TemplateView

# category labels
labels = ['confident', 'unconfident', 'handles pressure', 'cant handle pressure', 'interested',
          'uninsterested', 'happy', 'unhappy', 'friendly', 'unfriendly']

label_dict = dict(zip(labels, range(1, len(labels) + 1)))

# Load the pretrained model
model_path = 'C:/Users/Acer/Documents/coding/Upwork/Peter Gretale/models/h5/words/model_1000.h5'
model = load_model(model_path)


def preprocess(x, padding_shape=30):
    return np.array([ord(i.lower()) - ord('a')+1 if not i.isdigit() and i != ' ' else 0 for i in list(x)] + ([0] * (padding_shape - len(x))), dtype=int)


def predict(d: dict, s: str, model):
    token = preprocess(s)
    output = model.predict(np.array([token]))
    ID = int(tf.keras.backend.argmax(output))

    for k, v in d.items():
        if v == ID:
            return k

    return 'unclassified'


# Classification call class
class Classification(APIView):

    permission_classes = (AllowAny,)

    def post(self, request):
        try:
            # text to predict
            text = request.POST.get('word')

            # calling prediction function
            pred = predict(label_dict, text, model)
            print('Prediction: {}'.format(pred))

            return JsonResponse({'message': 'Prediction: ' + str(pred), 'code': 200, 'status': 'Success'})
        except Exception as e:
            print('email error:', e)
            return JsonResponse({'message': 'Something went wrong', 'code': 500, 'status': 'Error', 'error': str(e)})


class Home(TemplateView):
    template_name = ('home.html')
