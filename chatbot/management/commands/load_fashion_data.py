import os
import pandas as pd
import json
from django.core.management.base import BaseCommand
from django.core.files import File
from chatbot.models import ClothingItem
from chatbot.clip_utils import encode_text

class Command(BaseCommand):
    help = 'Loads fashion product images dataset'
    
    def handle(self, *args, **options):
        # UPDATE THIS PATH to match your actual extracted folder
        base_dir = 'C:\\Users\\DELL\\Documents\\StyleMatch_data\\fashion_product_images'
        csv_path = os.path.join(base_dir, 'styles.csv')
        images_dir = os.path.join(base_dir, 'images')
        
        # Check if paths exist
        if not os.path.exists(csv_path):
            self.stdout.write(self.style.ERROR(f'CSV file not found at: {csv_path}'))
            return
            
        if not os.path.exists(images_dir):
            self.stdout.write(self.style.ERROR(f'Images directory not found at: {images_dir}'))
            return
        
        # Read CSV with error handling
        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip', encoding='utf-8')
            self.stdout.write(f"✅ Successfully read CSV with {len(df)} rows")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Failed to read CSV: {e}"))
            return
        
        # Filter to about 100 items for diversity
        filtered_df = df.groupby('masterCategory').head(25)
        
        self.stdout.write(f"Processing {len(filtered_df)} items...")
        
        success_count = 0
        for index, row in filtered_df.iterrows():
            try:
                # Use the ready-made description from the dataset
                description = row['productDisplayName']
                
                # Generate embedding from the description
                text_embedding = encode_text(description)
                
                # Create database item
                item = ClothingItem(
                    description=description,
                    embedding=json.dumps(text_embedding.tolist())
                )
                
                # Find and save the corresponding image
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
        
        self.stdout.write(self.style.SUCCESS(f'Successfully loaded {success_count} items!'))