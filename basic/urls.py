from django.urls import path

from . import views

urlpatterns = [
    path('', views.indexView, name=''),
    path('api', views.apiView, name=''),
]