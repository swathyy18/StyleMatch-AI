from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import JSONParser, MultiPartParser
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.http import JsonResponse
import requests
import json
import numpy as np
import tempfile
import os
from .models import ClothingItem
from .clip_utils import encode_image, encode_text
from sklearn.metrics.pairwise import cosine_similarity

@method_decorator(csrf_exempt, name='dispatch')
class OutfitRecommendationView(APIView):
    parser_classes = [JSONParser, MultiPartParser]
    
    def post(self, request):
        text_input = request.data.get('text')
        image_input = request.FILES.get('image')
        
        # Case 1: User uploads image + text
        if image_input and text_input:
            return self.handle_image_with_text(image_input, text_input)
        
        # Case 2: User uploads only image
        elif image_input:
            return self.handle_image_only(image_input)
        
        # Case 3: User sends only text
        elif text_input:
            return self.handle_text_only(text_input)
        
        else:
            return Response({"error": "No text or image provided"}, status=400)
    
    def handle_image_with_text(self, image_file, user_text):
        """User uploads image + text like 'recommendations for this'"""
        try:
            # Step 1: Use CLIP to identify the image
            image_description = self.identify_image_with_clip(image_file)
            
            # NEW: Check if user is asking for shopping links
            if self.is_shopping_request(user_text):
                shopping_links = self.get_shopping_links(user_text, image_description)
                return Response({
                    "identified_item": image_description,
                    "user_request": user_text,
                    "shopping_links": shopping_links
                })
            
            # Step 2: Combine image info with user's text for LLM
            llm_prompt = f"Item: {image_description}. User request: '{user_text}'"
            
            # Step 3: Get LLM recommendation with specific context
            recommendation = self.get_llm_recommendation(llm_prompt, "image_with_text")
            
            return Response({
                "identified_item": image_description,
                "user_request": user_text,
                "recommendation": recommendation
            })
            
        except Exception as e:
            return Response({"error": f"Image processing failed: {str(e)}"}, status=500)
    
    def handle_image_only(self, image_file):
        """User uploads only image"""
        try:
            image_description = self.identify_image_with_clip(image_file)
            recommendation = self.get_llm_recommendation(image_description, "image_only")
            
            return Response({
                "identified_item": image_description,
                "recommendation": recommendation
            })
            
        except Exception as e:
            return Response({"error": f"Image processing failed: {str(e)}"}, status=500)
    
    def handle_text_only(self, user_text):
        """User sends only text"""
        print(f"DEBUG: User text: '{user_text}'")
        print(f"DEBUG: Is shopping request: {self.is_shopping_request(user_text)}")
        # NEW: Check if user is asking for shopping links
        if self.is_shopping_request(user_text):
            shopping_links = self.get_shopping_links(user_text)
            return Response({
                "user_request": user_text,
                "shopping_links": shopping_links
            })
        
        recommendation = self.get_llm_recommendation(user_text, "text")
        return Response({"recommendation": recommendation})
    
    def identify_image_with_clip(self, image_file):
        """Use CLIP to find the closest matching item in database"""
        # Save uploaded image to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            for chunk in image_file.chunks():
                tmp_file.write(chunk)
            tmp_path = tmp_file.name
        
        try:
            # Get CLIP embedding for uploaded image
            uploaded_embedding = encode_image(tmp_path)
            
            # Find closest match in database
            closest_item = self.find_closest_item(uploaded_embedding)
            
            return closest_item.description
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
    
    def find_closest_item(self, query_embedding):
        """Find the database item with closest embedding to query"""
        best_similarity = -1
        best_item = None
        
        for item in ClothingItem.objects.all():
            # Convert stored embedding back to numpy array
            item_embedding = np.array(json.loads(item.embedding))
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1), 
                item_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_item = item
        return best_item
    
    #SHOPPING LINKS PART

    def is_shopping_request(self, user_text):
        """Check if the user is asking for shopping links"""
        if not user_text:
            return False
    
        shopping_keywords = [
        'buy', 'purchase', 'shop', 'where to buy', 'where can i buy', 'get this',
        'amazon', 'flipkart', 'myntra', 'ajio', 'nykaa', 'meesho',
        'link', 'links', 'shopping', 'online', 'website', 'store',
        'shopping links', 'buy this', 'purchase this', 'shop for',
        'shopping sites', 'ecommerce', 'online store', 'product links for'
        ]

        user_text_lower = user_text.lower()

        # Check for exact matches and partial matches
        for keyword in shopping_keywords:
            if keyword in user_text_lower:
                return True

        return False

    def get_shopping_links(self, prompt, image_description=None):
        """Universal shopping links that work for any search term"""
        try:
            search_term = image_description if image_description else self.clean_shopping_text(prompt)
            encoded_term = search_term.replace(' ', '+')
        
            response = f"🛍️ **Shopping Results for '{search_term}':**\n\n"
        
            response += "**🔍 Smart Search Links:**\n\n"
        
            # Amazon links with different filters
            response += "**Amazon:**\n"
            response += f"• 🏆 **Best Rated**: https://www.amazon.in/s?k={encoded_term}&rh=p_72%3A1318476031&s=review-rank\n"
            response += f"• 🆕 **Latest Arrivals**: https://www.amazon.in/s?k={encoded_term}&s=date-desc-rank\n\n"
        
            # Flipkart links
            response += "**Flipkart:**\n"
            response += f"• 🏆 **Popular Choices**: https://www.flipkart.com/search?q={encoded_term}&sort=popularity\n"
            response += f"• ⭐ **4★+ Rated**: https://www.flipkart.com/search?q={encoded_term}&sort=relevance\n"
            response += f"• 💵 **Price Low to High**: https://www.flipkart.com/search?q={encoded_term}&sort=price_asc\n\n"
        
    
            return response
        
        except Exception as e:
            print(f"❌ Shopping links error: {e}")
            return f"🔍 Search for '{prompt}' on Amazon, Flipkart, or Myntra for great options!"

    def clean_shopping_text(self, prompt):
        """Clean the search query - keep only the product description"""
        if not prompt:
            return "fashion clothing"
    
        # Remove shopping-related phrases but keep the core item
        shopping_phrases = [
        'buy', 'purchase', 'shop', 'where to', 'where can i', 'get this',
        'links for', 'amazon', 'flipkart', 'myntra', 'meesho', 'ajio',
        'links', 'shopping', 'please', 'can you', 'could you', 'give me',
        'show me', 'i want', 'i need', 'for me', 'recommend', 'suggest',
        'product links for', 'shopping links for', 'where to find',
        'looking for', 'need to buy', 'want to purchase','ping'
        ]
    
        clean = prompt.lower()
    
        # Remove shopping phrases
        for phrase in shopping_phrases:
            clean = clean.replace(phrase, '')
    
        # Remove extra spaces and clean up
        clean = ' '.join(clean.split()).strip()
    
        # If empty after cleaning, use a default
        if not clean:
            clean = 'fashion clothing'
    
        return clean
    
    def get_llm_recommendation(self, prompt, context_type="text"):
        """Get fashion recommendations with context-aware prompts"""
        try:
            # Different prompts for different scenarios
            if context_type == "image_with_text":
                system_prompt = """You are a professional fashion stylist. Based on the clothing item described, provide 2 complete outfit suggestions.

FORMAT:
Outfit 1: [Occasion - e.g., Casual Day Out]
- Top: [specific suggestion]
- Bottom: [specific suggestion] 
- Footwear: [specific suggestion]
- Accessories: [2-3 specific items]
- Why it works: [brief explanation]

Outfit 2: [Different Occasion - e.g., Smart Casual]
- Top: [specific suggestion]
- Bottom: [specific suggestion]
- Footwear: [specific suggestion]
- Accessories: [2-3 specific items]
- Why it works: [brief explanation]

Be specific with colors, styles, and materials."""
                
                full_prompt = f"{system_prompt}\n\nItem: {prompt}"
                
            elif context_type == "image_only":
                system_prompt = """You are a fashion expert. For this clothing item, suggest 3 versatile ways to style it for different occasions.

Provide specific recommendations for:
1. Casual everyday wear
2. Smart casual/office appropriate  
3. Evening/date night

Include specific clothing items, colors, and styling tips."""
                
                full_prompt = f"{system_prompt}\n\nItem: {prompt}"
                
            else:  # text_only
                system_prompt = """You are a fashion consultant. Create complete outfit recommendations based on the user's request.

For each suggestion, include:
- Occasion/context
- Specific clothing items (be detailed)
- Color coordination tips
- Footwear and accessories
- Styling notes

Make it practical and fashionable."""
                
                full_prompt = f"{system_prompt}\n\nUser request: {prompt}"

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "gemma:2b",
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 600
                    }
                },
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                return self.clean_response(result['response'])
            else:
                return "I'd recommend focusing on fit, color coordination, and occasion-appropriate styling. Pair with complementary pieces that enhance your personal style."
                
        except requests.exceptions.ConnectionError:
            return "I apologize, but I'm having trouble connecting to the fashion recommendation service right now. Please try again later."
        
        except requests.exceptions.Timeout:
            return "The fashion recommendation service is taking longer than expected. Please try again in a moment."
            
        except Exception as e:
            return "For a stylish look, consider pairing with well-fitting complementary pieces, appropriate footwear, and accessories that match the occasion and your personal style."

    def clean_response(self, text):
        """Clean up the LLM response"""
        # Remove any repetitive phrases or markdown
        clean_text = text.split('###')[0]  # Remove markdown sections if any
        clean_text = clean_text.split('Note:')[0]  # Remove notes
        clean_text = clean_text.split('Remember:')[0]  # Remove reminders
        return clean_text.strip()
    
def chat_test_page(request):
    return render(request, "chat.html")