from transformers import pipeline
from PIL import Image
import torch
import sys

def analyze_food_image(image_path):
    # Load vision model for ingredient detection
    print("Loading vision model...")
    try:
        # Using a small model for demonstration
        vision_model = pipeline("image-classification", 
                               model="microsoft/resnet-50")
        
        # Load and process image
        print(f"Processing image: {image_path}")
        image = Image.open(image_path)
        result = vision_model(image, top_k=5)
        
        print("\nDetected items:")
        for item in result:
            print(f"- {item['label']} ({item['score']*100:.1f}%)")
        
        # Load text model for suggestions
        print("\nGenerating suggestions...")
        text_model = pipeline("text-generation", 
                             model="distilgpt2",
                             max_length=100)
        
        # Create a simple prompt
        prompt = f"This food looks like {', '.join([item['label'] for item in result])}. A healthy recipe with this would be:"
        response = text_model(prompt)[0]["generated_text"]
        
        print("\nRecipe suggestion:")
        print(response)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python demo.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    analyze_food_image(image_path)