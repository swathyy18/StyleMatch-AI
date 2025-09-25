from django.urls import path
from . import views

urlpatterns = [
    path('recommend/', views.OutfitRecommendationView.as_view(), name='outfit-recommend'),
    path('test/', views.chat_test_page, name='chat-test'),
]