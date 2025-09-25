from django.db import models

# Create your models here.
from django.db import models

class ClothingItem(models.Model):
    description = models.TextField()
    image = models.ImageField(upload_to='clothing_images/')
    embedding = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.description