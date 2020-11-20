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
model = load_model('/app/RESTapi/tfmodel.h5')


def get_embedding_vec(word):
    """
    return : embedding vector of a word
    """
    idx = word_embedding.dictionary.get(word.lower(), -1)

    if idx < 0:
        return np.zeros((EMBEDDING_DIM, ), dtype='float32')  # UNK

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


# Classification call class
class Classification(APIView):

    permission_classes = (AllowAny,)

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
            return JsonResponse({'message': 'Something went wrong', 'code': 500, 'status': 'Error', 'error': str(e)})


class Home(TemplateView):
    template_name = ('home.html')
