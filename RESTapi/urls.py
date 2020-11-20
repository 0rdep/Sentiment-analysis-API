from django.urls import path
from .views import *

urlpatterns = [
    path('classification/', Classification.as_view(), name='classification'),
    path('home/', Home.as_view(), name='home'),
]
