from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_image, name='upload-image'),
    # image_upload/urls.py
    path('clusters/', views.view_clusters, name='view-clusters'),
    path('upload-single/', views.upload_single_image, name='upload-single-image'),
    path('delete/<int:image_id>/', views.delete_image, name='delete-image'),
    path('delete-all/', views.delete_all_images, name='delete-all-images'),
]
