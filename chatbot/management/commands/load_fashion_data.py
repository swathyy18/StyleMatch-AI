import os
import pandas as pd
import json
from django.core.management.base import BaseCommand
from django.core.files import File
from chatbot.models import ClothingItem
from chatbot.clip_utils import encode_text

class Command(BaseCommand):
    help = 'Loads fashion product images dataset with clothing-only filtering'
    
    def handle(self, *args, **options):
        base_dir = 'C:\\Users\\DELL\\Documents\\StyleMatch_data\\fashion_product_images'
        csv_path = os.path.join(base_dir, 'styles.csv')
        images_dir = os.path.join(base_dir, 'images')
        
        if not os.path.exists(csv_path):
            self.stdout.write(self.style.ERROR(f'CSV file not found at: {csv_path}'))
            return
            
        if not os.path.exists(images_dir):
            self.stdout.write(self.style.ERROR(f'Images directory not found at: {images_dir}'))
            return
        
        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip', encoding='utf-8')
            self.stdout.write(f"✅ Successfully read CSV with {len(df)} rows")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Failed to read CSV: {e}"))
            return
        
        # FILTER OUT NON-CLOTHING ITEMS
        clothing_categories = [
            'Apparel', 'Clothing', 'Topwear', 'Bottomwear', 'Footwear', 
            'Dress', 'Innerwear', 'Socks', 'Loungewear and Nightwear'
        ]
        
        # Also exclude specific non-clothing categories
        exclude_categories = [
            'Watches', 'Perfume', 'Jewellery', 'Accessories', 'Personal Care',
            'Beauty and Personal Care', 'Home', 'Sports', 'Toys'
        ]
        
        # Filter for clothing only
        clothing_df = df[
            df['masterCategory'].isin(clothing_categories) | 
            df['subCategory'].isin(clothing_categories) |
            df['articleType'].str.contains('shirt|dress|pant|jean|top|skirt|jacket|shoe', case=False, na=False)
        ]
        
        # Exclude non-clothing items
        clothing_df = clothing_df[
            ~clothing_df['masterCategory'].isin(exclude_categories) &
            ~clothing_df['subCategory'].isin(exclude_categories) &
            ~clothing_df['articleType'].str.contains('watch|perfume|jewel', case=False, na=False)
        ]
        
        self.stdout.write(f"📊 After filtering: {len(clothing_df)} clothing items found")
        
        # EXCLUDE ITEMS ALREADY IN DATABASE
        existing_descriptions = set(ClothingItem.objects.values_list('description', flat=True))
        new_items_df = clothing_df[~clothing_df['productDisplayName'].isin(existing_descriptions)]
        
        self.stdout.write(f"🆕 New items to add: {len(new_items_df)} (excluding {len(existing_descriptions)} existing items)")
        
        if len(new_items_df) == 0:
            self.stdout.write(self.style.WARNING("No new items to add!"))
            return
        
        # Process new items
        success_count = 0
        for index, row in new_items_df.iterrows():
            try:
                description = row['productDisplayName']
                
                # Skip if description is too generic or short
                if len(description) < 10:
                    continue
                
                text_embedding = encode_text(description)
                
                item = ClothingItem(
                    description=description,
                    embedding=json.dumps(text_embedding.tolist())
                )
                
                image_filename = f"{row['id']}.jpg"
                image_path = os.path.join(images_dir, image_filename)
                
                if os.path.exists(image_path):
                    with open(image_path, 'rb') as f:
                        item.image.save(image_filename, File(f))
                    item.save()
                    success_count += 1
                    self.stdout.write(self.style.SUCCESS(f'[{success_count}] Added: {description}'))
                else:
                    self.stdout.write(self.style.WARNING(f'Image not found: {image_filename}'))
                    
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error with row {index}: {str(e)}'))
        
        self.stdout.write(self.style.SUCCESS(f'✅ Successfully loaded {success_count} new clothing items!'))