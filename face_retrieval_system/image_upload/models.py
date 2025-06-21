from django.db import models
from cloudinary.models import CloudinaryField

class UploadedImage(models.Model):
    image = CloudinaryField('image')
    public_id = models.CharField(max_length=255, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.public_id or self.image.url

class ImageEmbedding(models.Model):
    image = models.ForeignKey(UploadedImage, on_delete=models.CASCADE, related_name='embeddings')
    embedding = models.JSONField()  # Store embedding as JSON
    cluster_id = models.IntegerField(null=True, blank=True)


    def __str__(self):
        return f"Embedding for {self.image.public_id}"

    class Meta:
        indexes = [
            models.Index(fields=['image']),
        ]
