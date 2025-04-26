from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('main/', views.main, name='main'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('get_fatigue_score/', views.get_fatigue_score, name='get_fatigue_score'),
]
