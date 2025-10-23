from django.db import models
from django.contrib.auth.models import User
import json

class WardrobeItem(models.Model):
    CATEGORY_CHOICES = [
        ('top', 'Top'),
        ('bottom', 'Bottom'),
        ('dress', 'Dress'),
        ('shoes', 'Shoes'),
        ('accessories', 'Accessories'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='wardrobe/')
    description = models.CharField(max_length=255)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    embedding = models.TextField()  # CLIP embedding stored as JSON
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username}'s {self.description}"