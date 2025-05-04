import json
import os
import base64
import requests
from langchain.tools import tool
from PIL import Image
from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from io import BytesIO
from typing import List, Optional
import logging
logging.basicConfig(level=logging.INFO)

from transformers import pipeline, AutoProcessor, AutoModelForCausalLM
import torch

# Vision model for ingredient detection
def load_vision_model():
    # Using CLIP for image understanding
    vision_model = pipeline("image-classification", 
                           model="openai/clip-vit-base-patch32")
    return vision_model

# Text generation model for analysis and suggestions
def load_language_model():
    # Using Falcon-7B or another free text generation model
    text_model = pipeline("text-generation", 
                         model="tiiuae/falcon-7b-instruct",
                         torch_dtype=torch.bfloat16,
                         device_map="auto",
                         max_length=200,
                         do_sample=True)
    return text_model

# Function to detect ingredients from image
def detect_ingredients(image_path):
    vision_model = load_vision_model()
    image = Image.open(image_path)
    
    # Extract visual features and predict potential ingredients
    result = vision_model(image, top_k=10)
    # Process results to extract food-related labels
    ingredients = [item["label"] for item in result if "food" in item["label"] or "vegetable" in item["label"]]
    
    return ingredients

# Function to filter ingredients based on dietary restrictions
def filter_ingredients(ingredients, dietary_restrictions):
    text_model = load_language_model()
    
    prompt = f"""
    Given the following ingredients: {', '.join(ingredients)}
    And these dietary restrictions: {dietary_restrictions}
    
    List only the ingredients that comply with these restrictions:
    """
    
    response = text_model(prompt)[0]["generated_text"]
    # Extract the filtered ingredients from the response
    filtered_ingredients = parse_ingredients_from_response(response)
    
    return filtered_ingredients

# Function to suggest recipes
def generate_recipe_suggestions(ingredients, dietary_restrictions):
    text_model = load_language_model()
    
    prompt = f"""
    Create a healthy recipe using some or all of these ingredients: {', '.join(ingredients)}
    The recipe should be suitable for people with these dietary restrictions: {dietary_restrictions}
    
    Include:
    1. Recipe name
    2. Ingredients list
    3. Preparation steps
    4. Approximate nutrition facts
    """
    
    response = text_model(prompt)[0]["generated_text"]
    return response

# Helper function to parse ingredients from model response
def parse_ingredients_from_response(response):
    # Simple parsing logic - can be enhanced
    lines = response.split('\n')
    for i, line in enumerate(lines):
        if "ingredients" in line.lower():
            ingredients_text = lines[i+1]
            ingredients = [ing.strip() for ing in ingredients_text.split(',')]
            return ingredients
    
    # Fallback parsing
    return [item.strip() for item in response.split(',') if item.strip()]

logging.info("Extracting ingredients from image...")

credentials = Credentials(
                   url = "https://us-south.ml.cloud.ibm.com",
                   api_key = "<YOUR_API_KEY>" # Replace with your API key
                  )
client = APIClient(credentials)
project_id = "skills-network"


class ExtractIngredientsTool():
    @tool("Extract ingredients")
    def extract_ingredient(image_input: str):
        """
        Extract ingredients from a food item image.
        
        """
        if image_input.startswith("http"):  # Check if input is a URL
         
            response = requests.get(image_input)
            response.raise_for_status()
            image_bytes = BytesIO(response.content)
        else:
         
            if not os.path.isfile(image_input):
                raise FileNotFoundError(f"No file found at path: {image_input}")
            with open(image_input, "rb") as file:
                image_bytes = BytesIO(file.read())

        encoded_image = base64.b64encode(image_bytes.read()).decode("utf-8")

        model = ModelInference(
            model_id="meta-llama/llama-3-2-90b-vision-instruct",
            credentials=credentials,
            project_id=project_id,
            params={"max_tokens": 300},
        )
        response = model.chat(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract ingredients from the food item image"},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded_image}}
                    ],
                }
            ]
        )

        return response['choices'][0]['message']['content']


class FilterIngredientsTool:
    @tool("Filter ingredients")
    def filter_ingredients(raw_ingredients: str) -> List[str]:
        """
        Processes the raw ingredient data and filters out non-food items or noise.
        
        """
        ingredients = [ingredient.strip().lower() for ingredient in raw_ingredients.split(',') if ingredient.strip()]
        return ingredients

class DietaryFilterTool:
    @tool("Filter based on dietary restrictions")
    def filter_based_on_restrictions(ingredients: List[str], dietary_restrictions: Optional[str] = None) -> List[str]:
        """
        Uses an LLM model to filter ingredients based on dietary restrictions.

        """
        if not dietary_restrictions:
            return ingredients

        model = ModelInference(
            model_id="ibm/granite-3-8b-instruct",
            credentials=credentials,
            project_id=project_id,
            params={"max_tokens": 150},
        )

        prompt = f"""
        You are an AI nutritionist specialized in dietary restrictions. 
        Given the following list of ingredients: {', '.join(ingredients)}, 
        and the dietary restriction: {dietary_restrictions}, 
        remove any ingredient that does not comply with this restriction. 
        Return only the compliant ingredients as a comma-separated list with no additional commentary.
        """

        response = model.chat(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ],
                }
            ]
        )

        filtered = response['choices'][0]['message']['content'].strip().lower()
        filtered_list = [item.strip() for item in filtered.split(',') if item.strip()]
        return filtered_list

    
class NutrientAnalysisTool():
    @tool("Analyze nutritional values and calories of the dish from uploaded image")
    def analyze_image(image_input: str):
        """
        Provide a detailed nutrient breakdown and estimate the total calories of all ingredients from the uploaded image.
        
        """
        if image_input.startswith("http"):  # Check if input is a URL
            response = requests.get(image_input)
            response.raise_for_status()
            image_bytes = BytesIO(response.content)
        else:
            if not os.path.isfile(image_input):
                raise FileNotFoundError(f"No file found at path: {image_input}")
            with open(image_input, "rb") as file:
                image_bytes = BytesIO(file.read())

        encoded_image = base64.b64encode(image_bytes.read()).decode("utf-8")

        model = ModelInference(
            model_id="meta-llama/llama-3-2-90b-vision-instruct",
            credentials=credentials,
            project_id=project_id,
            params={"max_tokens": 300},
        )
        # Assistant prompt 
        assistant_prompt = """
            You are an expert nutritionist. Your task is to analyze the food items displayed in the image and provide a detailed nutritional assessment using the following format:
        1. **Identification**: List each identified food item clearly, one per line.
        2. **Portion Size & Calorie Estimation**: For each identified food item, specify the portion size and provide an estimated number of calories. Use bullet points with the following structure:
        - **[Food Item]**: [Portion Size], [Number of Calories] calories
        Example:
        *   **Salmon**: 6 ounces, 210 calories
        *   **Asparagus**: 3 spears, 25 calories
        3. **Total Calories**: Provide the total number of calories for all food items.
        Example:
        Total Calories: [Number of Calories]
        4. **Nutrient Breakdown**: Include a breakdown of key nutrients such as **Protein**, **Carbohydrates**, **Fats**, **Vitamins**, and **Minerals**. Use bullet points, and for each nutrient provide details about the contribution of each food item.
        Example:
        *   **Protein**: Salmon (35g), Asparagus (3g), Tomatoes (1g) = [Total Protein]
        5. **Health Evaluation**: Evaluate the healthiness of the meal in one paragraph.
        6. **Disclaimer**: Include the following exact text as a disclaimer:
        The nutritional information and calorie estimates provided are approximate and are based on general food data. 
        Actual values may vary depending on factors such as portion size, specific ingredients, preparation methods, and individual variations. 
        For precise dietary advice or medical guidance, consult a qualified nutritionist or healthcare provider.
        Format your response exactly like the template above to ensure consistency.
        """
        response = model.chat(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": assistant_prompt},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded_image}}
                    ],
                }
            ]
        )

        return response['choices'][0]['message']['content']