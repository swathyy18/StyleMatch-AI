from django.contrib import admin
from .models import WardrobeItem

@admin.register(WardrobeItem)
class WardrobeItemAdmin(admin.ModelAdmin):
    list_display = ['user', 'description', 'category', 'created_at']
    list_filter = ['category', 'created_at']