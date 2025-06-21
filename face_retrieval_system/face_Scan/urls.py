from django.urls import path
from . import views

urlpatterns = [
    path('scan/', views.scan_page, name='start_scan'),
    path('process/', views.process_scan, name='process_scan'),
]
