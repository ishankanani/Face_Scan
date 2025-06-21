from django import forms
from .models import EventImage

class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = EventImage
        fields = ['image']
        