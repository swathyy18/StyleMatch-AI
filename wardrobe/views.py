from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, JSONParser
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.http import HttpResponse
import json
import tempfile
import os
from .models import WardrobeItem
from chatbot.clip_utils import encode_image, encode_text
from chatbot.models import ClothingItem
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import re

@method_decorator(login_required, name='dispatch')
class WardrobeUploadView(APIView):
    parser_classes = [MultiPartParser, JSONParser]
    
    def post(self, request):
        print("🎯 Upload endpoint hit by user:", request.user)
        print("🎯 Files received:", list(request.FILES.keys()))
        
        images = request.FILES.getlist('images')
        print(f"🎯 Number of images: {len(images)}")
        
        if not images:
            print("❌ No images found in request")
            return Response({"error": "No images provided"}, status=400)
        
        uploaded_items = []
        
        for i, image in enumerate(images):
            try:
                print(f"🎯 Processing image {i+1}: {image.name}")
                
                # Save image to temporary file for CLIP processing
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    for chunk in image.chunks():
                        tmp_file.write(chunk)
                    tmp_path = tmp_file.name
                
                print(f"🎯 Temporary file created: {tmp_path}")
                
                # Get CLIP embedding and description
                try:
                    embedding = encode_image(tmp_path)
                    print("🎯 CLIP encoding successful")
                except Exception as e:
                    print(f"❌ CLIP encoding failed: {e}")
                    continue
                
                description = self.identify_item(embedding)
                category = self.detect_category(description)
                
                print(f"🎯 Identified: {description} → {category}")
                
                # Create wardrobe item
                item = WardrobeItem.objects.create(
                    user=request.user,
                    image=image,
                    description=description,
                    category=category,
                    embedding=json.dumps(embedding.tolist())
                )
                
                uploaded_items.append({
                    'id': item.id,
                    'description': description,
                    'category': category,
                    'image_url': item.image.url
                })
                
                print(f"✅ Successfully saved: {description}")
                
                # Cleanup
                os.unlink(tmp_path)
                
            except Exception as e:
                print(f"❌ Failed to process {image.name}: {str(e)}")
                import traceback
                traceback.print_exc()
                return Response({"error": f"Failed to process {image.name}: {str(e)}"}, status=500)
        
        print(f"✅ Upload completed: {len(uploaded_items)} items saved")
        return Response({
            "status": "success", 
            "uploaded_count": len(uploaded_items),
            "items": uploaded_items
        })
    
    def identify_item(self, embedding):
        """Match against ALL items in fashion database for maximum accuracy"""
        try:
            best_match = "fashion item"
            best_similarity = -1
            
            # Get ALL items from your fashion database - NO LIMIT
            print("🎯 Loading ALL database items for matching...")
            database_items = ClothingItem.objects.all()  # REMOVED LIMIT FOR MORE ITEMS
            total_items = database_items.count()
            print(f"🎯 Processing {total_items} database items")
            
            start_time = time.time()
            
            for i, db_item in enumerate(database_items):
                try:
                    # Convert stored embedding back to numpy
                    db_embedding = np.array(json.loads(db_item.embedding))
                    
                    similarity = cosine_similarity(
                        embedding.reshape(1, -1),
                        db_embedding.reshape(1, -1)
                    )[0][0]
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = db_item.description
                        
                    # Progress logging for large datasets
                    if (i + 1) % 200 == 0:
                        print(f"🎯 Processed {i+1}/{total_items} items...")
                        
                except Exception as e:
                    if i < 10:  # Only log first few errors to avoid spam
                        print(f"⚠️ Error processing database item {db_item.id}: {e}")
                    continue
            
            processing_time = time.time() - start_time
            print(f"✅ Database matching completed in {processing_time:.2f}s")
            print(f"🎯 Best match: {best_match} (similarity: {best_similarity:.3f})")
            
            # If similarity is decent, use database match
            if best_similarity > 0.15:
                return best_match
            else:
                print(f"⚠️ Low similarity ({best_similarity:.3f}), using fallback")
                return self.fallback_identify(embedding)
            
        except Exception as e:
            print(f"❌ Database matching failed: {e}")
            return self.fallback_identify(embedding)
    
    def fallback_identify(self, embedding):
        """Improved fallback method with Indian and Western fashion items"""
        fashion_items = {
            # Western Dresses
            "red dress": "a red dress",
            "black dress": "a black dress",
            "floral dress": "a floral print dress",
            "summer dress": "a light summer dress",
            "evening dress": "an elegant evening dress",
            "casual dress": "a casual everyday dress",
            
            # Western Tops
            "white t-shirt": "a white cotton t-shirt",
            "black t-shirt": "a black cotton t-shirt", 
            "blue shirt": "a blue shirt",
            "striped shirt": "a blue and white striped shirt",
            "blouse": "a feminine silk blouse",
            "sweater": "a cozy knit sweater",
            "tank top": "a simple tank top",
            
            # Western Bottoms
            "blue jeans": "blue denim jeans",
            "black jeans": "black denim jeans", 
            "denim skirt": "a denim skirt",
            "black skirt": "a black skirt",
            "leggings": "black leggings",
            "shorts": "denim shorts",
            "wide leg pants": "wide leg trousers",
            
            # Western Shoes
            "sneakers": "white athletic sneakers",
            "high heels": "black high heel shoes",
            "sandals": "leather sandals",
            "boots": "ankle boots",
            "flats": "ballet flats",
            
            # Western Outerwear
            "jacket": "a denim jacket",
            "blazer": "a formal blazer",
            "coat": "a winter coat",
            "cardigan": "a knit cardigan",
            
            # Indian Traditional - Kurtis & Tops
            "red kurti": "a red Indian kurti",
            "blue kurti": "a blue Indian kurti", 
            "green kurti": "a green Indian kurti",
            "yellow kurti": "a yellow Indian kurti",
            "pink kurti": "a pink Indian kurti",
            "printed kurti": "a printed Indian kurti",
            "embroidered kurti": "an embroidered Indian kurti",
            "anarkali kurti": "an anarkali style kurti",
            "long kurti": "a long Indian kurti",
            "short kurti": "a short Indian kurti",
            "kurti": "an Indian kurti top",
            
            # Indian Traditional - Sarees
            "silk saree": "a silk Indian saree",
            "cotton saree": "a cotton Indian saree",
            "banarasi saree": "a banarasi silk saree",
            "kanjeevaram saree": "a kanjeevaram silk saree",
            "printed saree": "a printed Indian saree",
            "embroidered saree": "an embroidered Indian saree",
            "georgette saree": "a georgette Indian saree",
            "chiffon saree": "a chiffon Indian saree",
            "saree": "an Indian saree",
            
            # Indian Traditional - Bottoms
            "leggings": "black leggings",
            "palazzo pants": "flowy palazzo pants",
            "churidar": "a churidar bottom",
            "dhoti pants": "dhoti style pants",
            "salwar": "a salwar bottom",
            "patiala": "a patiala salwar",
            
            # Indian Traditional - Dupattas & Accessories
            "dupatta": "a matching dupatta",
            "printed dupatta": "a printed dupatta",
            "embroidered dupatta": "an embroidered dupatta",
            "silver jewelry": "silver Indian jewelry",
            "gold jewelry": "gold Indian jewelry",
            
            # Indian Footwear
            "juttis": "traditional Indian juttis",
            "mojaris": "traditional Indian mojaris",
            "kolhapuris": "traditional Kolhapuri sandals",
            "ethnic sandals": "ethnic Indian sandals"
        }
        
        best_match = "clothing item"
        best_similarity = -1
        
        for desc, prompt in fashion_items.items():
            try:
                text_embed = encode_text(prompt)
                similarity = cosine_similarity(
                    embedding.reshape(1, -1), 
                    text_embed.reshape(1, -1)
                )[0][0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = desc
            except Exception as e:
                continue
        
        print(f"🎯 Fallback match: {best_match} (similarity: {best_similarity:.3f})")
        return best_match

    def detect_category(self, description):
        """Improved category detection with better priority for Indian clothing"""
        description_lower = description.lower()
        
        # Priority 1: Check for Indian traditional wear - Sarees (most specific)
        saree_keywords = ['saree', 'sari', 'banarasi', 'kanjeevaram', 'georgette', 'chiffon']
        if any(keyword in description_lower for keyword in saree_keywords):
            return 'saree'
        
        # Priority 2: Check for Indian traditional wear - Kurtis (before general tops)
        kurti_keywords = ['kurti', 'kurta', 'anarkali', 'kurtis', 'kurtas']
        if any(keyword in description_lower for keyword in kurti_keywords):
            return 'kurti'
        
        # Priority 3: Check for shoes/footwear 
        shoe_keywords = ['shoe', 'sandal', 'heel', 'sneaker', 'boot', 'pump', 'loafer', 'flat', 'juttis', 'mojaris', 'kolhapuris']
        if any(keyword in description_lower for keyword in shoe_keywords):
            return 'shoes'
        
        # Priority 4: Check for Indian traditional bottoms (before western bottoms)
        indian_bottom_keywords = ['palazzo', 'churidar', 'dhoti', 'salwar', 'patiala', 'leggings']
        if any(keyword in description_lower for keyword in indian_bottom_keywords):
            return 'indian_bottom'
        
        # Priority 5: Check for Western dresses
        dress_keywords = ['dress', 'gown', 'jumpsuit', 'maxi', 'midi']
        if any(keyword in description_lower for keyword in dress_keywords):
            return 'dress'
        
        # Priority 6: Check for dupattas
        dupatta_keywords = ['dupatta', 'stole', 'scarf']
        if any(keyword in description_lower for keyword in dupatta_keywords):
            return 'dupatta'
        
        # Priority 7: Check for Western bottoms
        bottom_keywords = ['pant', 'jean', 'trouser', 'short', 'jogger', 'skirt']
        if any(keyword in description_lower for keyword in bottom_keywords):
            return 'bottom'
        
        # Priority 8: Check for Western tops (last to avoid catching kurtis)
        top_keywords = ['shirt', 'top', 'blouse', 't-shirt', 'tank', 'crop top', 'sweater', 'hoodie', 'blazer', 'jacket', 'cardigan']
        if any(keyword in description_lower for keyword in top_keywords):
            return 'top'
        
        # Priority 9: Check for Indian jewelry and accessories
        jewelry_keywords = ['jewelry', 'jewellery', 'necklace', 'earring', 'bangle', 'bracelet']
        if any(keyword in description_lower for keyword in jewelry_keywords):
            return 'accessories'
        
        return 'accessories'

@method_decorator(login_required, name='dispatch')
class GenerateOutfitsView(APIView):
    def post(self, request):
        try:
            user_items = WardrobeItem.objects.filter(user=request.user)
            print(f"🎯 Generating outfits for {user_items.count()} items")
            
            if user_items.count() < 2:
                return Response({"error": "Need at least 2 items to generate outfits"}, status=400)
            
            # Debug: print all items and their categories/colors
            for item in user_items:
                color = self.extract_color_from_description(item.description)
                print(f"🎯 Item: {item.description} → Category: {item.category} → Color: {color}")
            
            outfits = self.generate_combinations(user_items)
            print(f"🎯 Generated {len(outfits)} outfit combinations")
            
            if not outfits:
                return Response({
                    "error": "Could not generate complete outfits. Try adding more diverse clothing items.",
                    "debug_info": {
                        "total_items": user_items.count(),
                        "categories": {
                            'western_tops': user_items.filter(category='top').count(),
                            'western_bottoms': user_items.filter(category='bottom').count(),
                            'western_dresses': user_items.filter(category='dress').count(),
                            'kurtis': user_items.filter(category='kurti').count(),
                            'sarees': user_items.filter(category='saree').count(),
                            'indian_bottoms': user_items.filter(category='indian_bottom').count(),
                            'dupattas': user_items.filter(category='dupatta').count(),
                            'shoes': user_items.filter(category='shoes').count(),
                            'accessories': user_items.filter(category='accessories').count()
                        }
                    }
                }, status=400)
            
            return Response({"outfits": outfits})
        
        except Exception as e:
            print(f"❌ Error in GenerateOutfitsView: {str(e)}")
            import traceback
            traceback.print_exc()
            return Response({"error": f"Internal server error: {str(e)}"}, status=500)
    
    def generate_combinations(self, items):
        """Generate valid outfit combinations with extensive color theory"""
        # Categorize items
        western_tops = [item for item in items if item.category == 'top']
        western_bottoms = [item for item in items if item.category == 'bottom']
        western_dresses = [item for item in items if item.category == 'dress']
        kurtis = [item for item in items if item.category == 'kurti']
        sarees = [item for item in items if item.category == 'saree']
        indian_bottoms = [item for item in items if item.category == 'indian_bottom']
        dupattas = [item for item in items if item.category == 'dupatta']
        shoes = [item for item in items if item.category == 'shoes']
        accessories = [item for item in items if item.category == 'accessories']
        
        print(f"🎯 Categorized: {len(western_tops)} western_tops, {len(western_bottoms)} western_bottoms, {len(western_dresses)} western_dresses")
        print(f"🎯 Indian wear: {len(kurtis)} kurtis, {len(sarees)} sarees, {len(indian_bottoms)} indian_bottoms, {len(dupattas)} dupattas")
        
        combinations = []
        
        # Strategy 1: Western Dress outfits with color-coordinated shoes
        if western_dresses:
            for dress in western_dresses[:3]:
                dress_color = self.extract_color_from_description(dress.description)
                matching_shoes = self.get_highly_matching_shoes(dress_color, shoes)
                
                if matching_shoes:
                    for shoe in matching_shoes[:2]:
                        combo = {
                            'type': 'western_dress_outfit',
                            'items': [
                                {
                                    'id': dress.id,
                                    'description': dress.description,
                                    'category': dress.category,
                                    'image': dress.image.url
                                },
                                {
                                    'id': shoe.id,
                                    'description': shoe.description,
                                    'category': shoe.category,
                                    'image': shoe.image.url
                                }
                            ],
                            'description': f"{dress.description} with {shoe.description}"
                        }
                        combinations.append(combo)
        
        # Strategy 2: Western Top + Bottom combinations with extensive color theory
        if western_tops and western_bottoms:
            for top in western_tops[:5]:
                top_color = self.extract_color_from_description(top.description)
                # Get highly matching bottoms only
                highly_matching_bottoms = self.get_highly_matching_bottoms(top_color, western_bottoms)
                
                for bottom in highly_matching_bottoms[:2]:
                    bottom_color = self.extract_color_from_description(bottom.description)
                    # Get shoes that match both top and bottom
                    highly_matching_shoes = self.get_highly_matching_shoes_for_outfit(top_color, bottom_color, shoes)
                    
                    if highly_matching_shoes:
                        for shoe in highly_matching_shoes[:1]:
                            combo = {
                                'type': 'western_top_bottom_outfit',
                                'items': [
                                    {
                                        'id': top.id,
                                        'description': top.description,
                                        'category': top.category,
                                        'image': top.image.url
                                    },
                                    {
                                        'id': bottom.id,
                                        'description': bottom.description,
                                        'category': bottom.category,
                                        'image': bottom.image.url
                                    },
                                    {
                                        'id': shoe.id,
                                        'description': shoe.description,
                                        'category': shoe.category,
                                        'image': shoe.image.url
                                    }
                                ],
                                'description': f"{top.description} with {bottom.description} and {shoe.description}"
                            }
                            combinations.append(combo)
                    else:
                        # Only create outfit if colors are highly compatible
                        if self.are_colors_highly_compatible(top_color, bottom_color):
                            combo = {
                                'type': 'western_top_bottom_outfit',
                                'items': [
                                    {
                                        'id': top.id,
                                        'description': top.description,
                                        'category': top.category,
                                        'image': top.image.url
                                    },
                                    {
                                        'id': bottom.id,
                                        'description': bottom.description,
                                        'category': bottom.category,
                                        'image': bottom.image.url
                                    }
                                ],
                                'description': f"{top.description} with {bottom.description}"
                            }
                            combinations.append(combo)
        
        # Strategy 3: Indian Kurti + Pants combinations (NO SKIRTS)
        if kurtis:
            for kurti in kurtis[:4]:
                kurti_color = self.extract_color_from_description(kurti.description)
                
                # Try Indian pants first (excluding skirts)
                highly_matching_pants = self.get_highly_matching_indian_bottoms(kurti_color, indian_bottoms)
                
                # If no Indian pants, try Western pants as fallback (also excluding skirts)
                if not highly_matching_pants:
                    western_pants_only = [b for b in western_bottoms if 'skirt' not in b.description.lower()]
                    highly_matching_pants = self.get_highly_matching_bottoms(kurti_color, western_pants_only)
                
                # Only create outfits if we have matching PANTS (no skirts)
                for pants in highly_matching_pants[:2]:
                    # Add dupatta if available and color-coordinated
                    highly_matching_dupattas = self.get_highly_matching_dupattas(kurti_color, dupattas)
                    
                    combo_items = [
                        {
                            'id': kurti.id,
                            'description': kurti.description,
                            'category': kurti.category,
                            'image': kurti.image.url
                        },
                        {
                            'id': pants.id,
                            'description': pants.description,
                            'category': pants.category,
                            'image': pants.image.url
                        }
                    ]
                    
                    if highly_matching_dupattas:
                        dupatta = highly_matching_dupattas[0]
                        combo_items.append({
                            'id': dupatta.id,
                            'description': dupatta.description,
                            'category': dupatta.category,
                            'image': dupatta.image.url
                        })
                    
                    # Add Indian footwear
                    indian_shoes = [s for s in shoes if self.is_indian_footwear(s.description)]
                    highly_matching_indian_shoes = self.get_highly_matching_shoes(kurti_color, indian_shoes)
                    
                    if highly_matching_indian_shoes:
                        shoe = highly_matching_indian_shoes[0]
                        combo_items.append({
                            'id': shoe.id,
                            'description': shoe.description,
                            'category': shoe.category,
                            'image': shoe.image.url
                        })
                    
                    combo = {
                        'type': 'indian_kurti_outfit',
                        'items': combo_items,
                        'description': f"{kurti.description} with {pants.description}" + (f" and {dupatta.description}" if highly_matching_dupattas else "")
                    }
                    combinations.append(combo)
        
        # Strategy 4: Saree outfits with color-coordinated blouses
        if sarees:
            for saree in sarees[:3]:
                saree_color = self.extract_color_from_description(saree.description)
                # Get highly matching blouses
                highly_matching_blouses = self.get_highly_matching_blouses(saree_color, western_tops)
                
                combo_items = [
                    {
                        'id': saree.id,
                        'description': saree.description,
                        'category': saree.category,
                        'image': saree.image.url
                    }
                ]
                
                if highly_matching_blouses:
                    blouse = highly_matching_blouses[0]
                    combo_items.append({
                        'id': blouse.id,
                        'description': blouse.description,
                        'category': blouse.category,
                        'image': blouse.image.url
                    })
                
                # Add Indian footwear
                indian_shoes = [s for s in shoes if self.is_indian_footwear(s.description)]
                highly_matching_indian_shoes = self.get_highly_matching_shoes(saree_color, indian_shoes)
                
                if highly_matching_indian_shoes:
                    shoe = highly_matching_indian_shoes[0]
                    combo_items.append({
                        'id': shoe.id,
                        'description': shoe.description,
                        'category': shoe.category,
                        'image': shoe.image.url
                    })
                
                combo = {
                    'type': 'saree_outfit',
                    'items': combo_items,
                    'description': f"{saree.description}" + (f" with {blouse.description}" if highly_matching_blouses else "")
                }
                combinations.append(combo)
        
        # Remove duplicates and return
        unique_combinations = []
        seen_combinations = set()
        
        for combo in combinations:
            combo_key = tuple(sorted(item['id'] for item in combo['items']))
            if combo_key not in seen_combinations:
                seen_combinations.add(combo_key)
                unique_combinations.append(combo)
        
        return unique_combinations[:10]

    def extract_color_from_description(self, description):
        """Extract color from description"""
        description_lower = description.lower()
        
        # Common colors mapping
        color_keywords = {
            'black': ['black', 'ebony', 'onyx'],
            'white': ['white', 'ivory', 'cream', 'off-white'],
            'red': ['red', 'crimson', 'scarlet', 'burgundy', 'maroon'],
            'blue': ['blue', 'navy', 'denim', 'sky blue', 'royal blue', 'light blue'],
            'green': ['green', 'emerald', 'olive', 'forest', 'mint'],
            'yellow': ['yellow', 'gold', 'mustard', 'lemon'],
            'pink': ['pink', 'rose', 'fuchsia', 'hot pink'],
            'purple': ['purple', 'violet', 'lavender', 'lilac'],
            'orange': ['orange', 'coral', 'peach'],
            'brown': ['brown', 'tan', 'beige', 'khaki', 'taupe'],
            'gray': ['gray', 'grey', 'charcoal', 'silver'],
            'multicolor': ['floral', 'print', 'pattern', 'striped', 'checkered', 'polka dot']
        }
        
        for color, keywords in color_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                return color
        
        return 'unknown'

    def get_highly_matching_bottoms(self, top_color, bottoms):
        """Return only bottoms that HIGHLY match with the top based on extensive color theory"""
        highly_matching = []
        moderately_matching = []
        
        for bottom in bottoms:
            bottom_color = self.extract_color_from_description(bottom.description)
            
            # Check for high compatibility
            if self.are_colors_highly_compatible(top_color, bottom_color):
                highly_matching.append(bottom)
            # Fallback to moderate matches
            elif self.colors_match(top_color, bottom_color):
                moderately_matching.append(bottom)
        
        # Return highly matching first, then moderately matching
        return highly_matching + moderately_matching[:2]

    def get_highly_matching_indian_bottoms(self, kurti_color, bottoms):
        """Return Indian pants (not skirts) that HIGHLY match with kurti"""
        highly_matching = []
        moderately_matching = []
        
        for bottom in bottoms:
            bottom_description = bottom.description.lower()
            
            # EXCLUDE skirts - only include pants-like items
            skirt_keywords = ['skirt', 'ghagra', 'lehenga', 'gown', 'dress']
            if any(skirt_word in bottom_description for skirt_word in skirt_keywords):
                continue  # Skip skirts completely
                
            bottom_color = self.extract_color_from_description(bottom.description)
            
            # For Indian wear, check for high compatibility
            if self.are_colors_highly_compatible(kurti_color, bottom_color):
                highly_matching.append(bottom)
            # Fallback to traditional Indian color combinations
            elif self.colors_complement(kurti_color, bottom_color):
                moderately_matching.append(bottom)
        
        return highly_matching + moderately_matching[:2]

    def get_highly_matching_shoes(self, item_color, shoes):
        """Return shoes that HIGHLY match with the item"""
        highly_matching = []
        neutral_matching = []
        
        for shoe in shoes:
            shoe_color = self.extract_color_from_description(shoe.description)
            
            # High compatibility
            if self.are_colors_highly_compatible(item_color, shoe_color):
                highly_matching.append(shoe)
            # Neutral shoes that always work
            elif shoe_color in ['black', 'white', 'brown', 'nude']:
                neutral_matching.append(shoe)
        
        return highly_matching + neutral_matching[:2]

    def get_highly_matching_shoes_for_outfit(self, top_color, bottom_color, shoes):
        """Return shoes that HIGHLY match with both top and bottom"""
        highly_matching = []
        neutral_matching = []
        
        for shoe in shoes:
            shoe_color = self.extract_color_from_description(shoe.description)
            
            # Check if shoe color works well with both top and bottom
            if (self.are_colors_highly_compatible(top_color, shoe_color) and 
                self.are_colors_highly_compatible(bottom_color, shoe_color)):
                highly_matching.append(shoe)
            # Neutral shoes
            elif shoe_color in ['black', 'white', 'brown', 'nude']:
                neutral_matching.append(shoe)
        
        return highly_matching + neutral_matching[:2]

    def get_highly_matching_dupattas(self, kurti_color, dupattas):
        """Return dupattas that HIGHLY match with kurti"""
        highly_matching = []
        
        for dupatta in dupattas:
            dupatta_color = self.extract_color_from_description(dupatta.description)
            
            # Dupattas should either match or beautifully contrast
            if (self.are_colors_highly_compatible(kurti_color, dupatta_color) or
                self.are_colors_beautiful_contrast(kurti_color, dupatta_color)):
                highly_matching.append(dupatta)
        
        return highly_matching[:2]

    def get_highly_matching_blouses(self, saree_color, tops):
        """Return blouses that HIGHLY match with saree"""
        highly_matching = []
        
        for top in tops:
            top_color = self.extract_color_from_description(top.description)
            
            # Blouse should either match or complement the saree
            if (self.are_colors_highly_compatible(saree_color, top_color) or
                self.are_colors_beautiful_contrast(saree_color, top_color)):
                highly_matching.append(top)
        
        return highly_matching[:2]

    def are_colors_highly_compatible(self, color1, color2):
        """Check if colors are highly compatible based on fashion color theory"""
        if color1 == 'unknown' or color2 == 'unknown':
            return False  # Be strict about unknown colors
        
        # Perfect matches
        if color1 == color2:
            return True
        
        # Classic fashion combinations
        classic_combinations = {
            'black': ['white', 'red', 'pink', 'yellow', 'gold', 'silver'],
            'white': ['black', 'navy', 'red', 'green', 'blue', 'pink', 'purple'],
            'navy': ['white', 'pink', 'red', 'yellow', 'light blue'],
            'red': ['white', 'black', 'navy', 'gray', 'khaki'],
            'blue': ['white', 'navy', 'gray', 'pink', 'yellow'],
            'pink': ['white', 'black', 'gray', 'navy', 'green'],
            'green': ['white', 'black', 'brown', 'khaki', 'navy'],
            'yellow': ['white', 'black', 'gray', 'navy', 'purple'],
            'purple': ['white', 'black', 'gray', 'yellow', 'pink'],
            'gray': ['white', 'black', 'pink', 'yellow', 'red', 'navy'],
            'brown': ['white', 'blue', 'green', 'pink', 'cream'],
            'khaki': ['white', 'navy', 'green', 'red', 'pink']
        }
        
        # Check both directions
        if color1 in classic_combinations and color2 in classic_combinations[color1]:
            return True
        if color2 in classic_combinations and color1 in classic_combinations[color2]:
            return True
        
        return False

    def are_colors_beautiful_contrast(self, color1, color2):
        """Check if colors create a beautiful contrast (for Indian wear especially)"""
        beautiful_contrasts = {
            'red': ['green', 'gold', 'yellow'],
            'green': ['red', 'pink', 'orange', 'purple'],
            'blue': ['orange', 'pink', 'gold'],
            'pink': ['green', 'blue', 'purple'],
            'purple': ['yellow', 'pink', 'gold'],
            'yellow': ['purple', 'red', 'blue'],
            'orange': ['blue', 'green', 'purple']
        }
        
        if color1 in beautiful_contrasts and color2 in beautiful_contrasts[color1]:
            return True
        if color2 in beautiful_contrasts and color1 in beautiful_contrasts[color2]:
            return True
        
        return False

    def colors_match(self, color1, color2):
        """Basic color matching (less strict fallback)"""
        if color1 == 'unknown' or color2 == 'unknown':
            return True
        
        neutral_colors = ['black', 'white', 'gray', 'brown', 'beige', 'khaki', 'navy']
        if color1 in neutral_colors or color2 in neutral_colors:
            return True
        
        return color1 == color2

    def colors_complement(self, color1, color2):
        """Check if colors complement each other for Indian wear"""
        if color1 == 'unknown' or color2 == 'unknown':
            return True
        
        indian_color_combinations = {
            'red': ['green', 'gold', 'yellow', 'pink'],
            'green': ['red', 'pink', 'orange', 'purple'],
            'blue': ['orange', 'pink', 'silver', 'yellow'],
            'pink': ['green', 'blue', 'purple', 'red'],
            'purple': ['pink', 'yellow', 'gold', 'green'],
            'yellow': ['purple', 'red', 'blue', 'green'],
            'orange': ['blue', 'green', 'purple', 'pink']
        }
        
        if color1 in indian_color_combinations and color2 in indian_color_combinations[color1]:
            return True
        if color2 in indian_color_combinations and color1 in indian_color_combinations[color2]:
            return True
        
        return False

    def is_indian_footwear(self, description):
        """Check if footwear is Indian style"""
        indian_footwear_keywords = ['juttis', 'mojaris', 'kolhapuris', 'ethnic']
        return any(keyword in description.lower() for keyword in indian_footwear_keywords)

@method_decorator(login_required, name='dispatch')
class WardrobeListView(APIView):
    def get(self, request):
        items = WardrobeItem.objects.filter(user=request.user)
        item_data = []
        for item in items:
            item_data.append({
                'id': item.id,
                'description': item.description,
                'category': item.category,
                'image_url': item.image.url,
                'created_at': item.created_at.strftime('%Y-%m-%d')
            })
        return Response({"items": item_data})

@method_decorator(login_required, name='dispatch')
class DeleteWardrobeItemView(APIView):
    def post(self, request):
        item_id = request.data.get('item_id')
        try:
            item = WardrobeItem.objects.get(id=item_id, user=request.user)
            item.delete()
            return Response({"status": "success", "message": "Item deleted successfully"})
        except WardrobeItem.DoesNotExist:
            return Response({"error": "Item not found"}, status=404)

# Regular Django view for the wardrobe page
@login_required
def wardrobe_page(request):
    return render(request, 'wardrobe/wardrobe.html')