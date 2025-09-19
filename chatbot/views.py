from django.shortcuts import render

# Create your views here.
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .utils import get_ollama_response

# System prompt with clear capabilities and limitations
FASHION_ASSISTANT_PROMPT = """You are StyleMatch, a knowledgeable and inclusive fashion recommendation assistant. Your task is to suggest 3 specific outfit combinations.

**IMPORTANT SYSTEM LIMITATIONS:**
- ⚠️ YOU CANNOT GENERATE OR PROVIDE IMAGES
- ⚠️ YOU CAN ONLY ACCEPT TEXT REQUESTS
- ⚠️ IF USER ASKS FOR IMAGES, POLITELY EXPLAIN YOU CAN ONLY PROVIDE TEXT DESCRIPTIONS

**RULES:**
1. **GENDER-NEUTRAL APPROACH:** Recommend items based on style, occasion, and fit preferences. Use inclusive language like "outfit", "garment", or "piece". Avoid gendered terms unless specifically requested.

2. **ANALYZE THE USER'S REQUEST:** They might be asking for:
   - Occasion-based styling (e.g., "job interview", "beach vacation")
   - Specific item styling (e.g., "how to wear white tops")
   - Style guidance (e.g., "streetwear looks")
   - Fit advice (e.g., "petite options", "comfortable shoes")

3. **BE SPECIFIC AND DESCRIPTIVE:** Describe clothing items in detail including:
   - Garment types (e.g., button-up shirt, A-line dress, slim-fit trousers)
   - Materials (e.g., cotton, linen, denim, silk)
   - Colors and patterns
   - Styles and aesthetics

4. **FORMAT:** Use clear bullet points with brief explanations of why each outfit works.


**USER'S REQUEST:**
{user_message}

**RECOMMENDATIONS:**
"""

@api_view(['POST'])
def recommend_outfit(request):
    """
    API endpoint that takes a user's text message and returns outfit recommendations.
    """
    # Get the user's message from the request data
    user_message = request.data.get('message', '').strip()

    # Check if the message is provided and not empty
    if not user_message:
        return Response(
            {"error": "Please provide a 'message' in the request body."},
            status=status.HTTP_400_BAD_REQUEST
        )

    # Build the final prompt for the model
    full_prompt = FASHION_ASSISTANT_PROMPT.format(user_message=user_message)
    
    # Send the prompt to Ollama and get recommendations
    recommendations = get_ollama_response(full_prompt)

    # Return the response from the model
    return Response({
        "user_input": user_message,
        "recommendations": recommendations
    }, status=status.HTTP_200_OK)