from django.urls import path
from . import views

urlpatterns = [
    path('', views.wardrobe_page, name='wardrobe_page'),
    path('api/upload/', views.WardrobeUploadView.as_view(), name='wardrobe_upload'),
    path('api/items/', views.WardrobeListView.as_view(), name='wardrobe_list'),
    path('api/generate-outfits/', views.GenerateOutfitsView.as_view(), name='generate_outfits'),
    path('api/delete-item/', views.DeleteWardrobeItemView.as_view(), name='delete_item'),
]