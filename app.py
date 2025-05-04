import gradio as gr
import base64
import time
from src.crew import NutriCoachRecipeCrew, NutriCoachAnalysisCrew
from transformers import pipeline
from PIL import Image
import torch
import sys
import argparse
import os

def format_recipe_output(final_output):
    """
    Formats the recipe output into a table-based Markdown format.
    
    """
    output = "## 🍽 Recipe Ideas\n\n"
    recipes = []

    if "recipes" in final_output:
        recipes = final_output["recipes"]
    else:
      
        recipe_task_output = final_output.get("recipe_suggestion_task")
        if recipe_task_output and hasattr(recipe_task_output, "json_dict") and recipe_task_output.json_dict:
            recipes = recipe_task_output.json_dict.get("recipes", [])
    
    if recipes:
        for idx, recipe in enumerate(recipes, 1):
            output += f"### {idx}. {recipe['title']}\n\n"
            
            # Create a table for ingredients
            output += "**Ingredients:**\n"
            output += "| Ingredient |\n"
            output += "|------------|\n"
            for ingredient in recipe['ingredients']:
                output += f"| {ingredient} |\n"
            output += "\n"
            
            # Display instructions and calorie estimate
            output += f"**Instructions:**\n{recipe['instructions']}\n\n"
            output += f"**Calorie Estimate:** {recipe['calorie_estimate']} kcal\n\n"
            output += "---\n\n"
    else:
        output += "No recipes could be generated."
    
    return output

def format_analysis_output(final_output):
    """
    Formats nutritional analysis output into a table-based Markdown format,
    including health evaluation at the end.
    
    """
    output = "## 🥗 Nutritional Analysis\n\n"
    
    if dish := final_output.get('dish'):
        output += f"**Dish:** {dish}\n\n"
    if portion := final_output.get('portion_size'):
        output += f"**Portion Size:** {portion}\n\n"
    if est_cal := final_output.get('estimated_calories'):
        output += f"**Estimated Calories:** {est_cal} calories\n\n"
    if total_cal := final_output.get('total_calories'):
        output += f"**Total Calories:** {total_cal} calories\n\n"

    output += "**Nutrient Breakdown:**\n\n"
    output += "| **Nutrient**       | **Amount** |\n"
    output += "|--------------------|------------|\n"
    
    nutrients = final_output.get('nutrients', {})
 
    for macro in ['protein', 'carbohydrates', 'fats']:
        if value := nutrients.get(macro):
            output += f"| **{macro.capitalize()}** | {value} |\n"
    
 
    vitamins = nutrients.get('vitamins', [])
    if vitamins:
        output += "\n**Vitamins:**\n\n"
        output += "| **Vitamin** | **%DV** |\n"
        output += "|-------------|--------|\n"
        for v in vitamins:
            name = v.get('name', 'N/A')
            dv = v.get('percentage_dv', 'N/A')
            output += f"| {name} | {dv} |\n"
    
    minerals = nutrients.get('minerals', [])
    if minerals:
        output += "\n**Minerals:**\n\n"
        output += "| **Mineral** | **Amount** |\n"
        output += "|-------------|-----------|\n"
        for m in minerals:
            name = m.get('name', 'N/A')
            amount = m.get('amount', 'N/A')
            output += f"| {name} | {amount} |\n"
    
    if health_eval := final_output.get('health_evaluation'):
        output += "\n**Health Evaluation:**\n\n"
        output += health_eval + "\n"
    
    return output


def analyze_food(image, dietary_restrictions, workflow_type, progress=gr.Progress(track_tqdm=True)):
    """
    Wrapper function for the Gradio interface.
    
    """
    
    image.save("uploaded_image.jpg")  
    image_path = "uploaded_image.jpg"

    inputs = {
        'uploaded_image': image_path,
        'dietary_restrictions': dietary_restrictions,
        'workflow_type': workflow_type
    }
    
    # Initialize the required crew instance based on workflow type
    if workflow_type == "recipe":
        crew_instance = NutriCoachRecipeCrew(
            image_data=image_path,
            dietary_restrictions=dietary_restrictions
        )
    elif workflow_type == "analysis":
        crew_instance = NutriCoachAnalysisCrew(
            image_data=image_path
        )
    else:
        return "Invalid workflow type. Choose 'recipe' or 'analysis'."

    # Run the crew workflow and get the result
    crew_obj = crew_instance.crew()
    final_output = crew_obj.kickoff(inputs=inputs)

    final_output = final_output.to_dict()

    if workflow_type == "recipe":
        recipe_markdown = format_recipe_output(final_output)
        return recipe_markdown
    elif workflow_type == "analysis":
        nutrient_markdown = format_analysis_output(final_output)
        return nutrient_markdown
    
    
css = """
.title {
    font-size: 1.5em !important; 
    text-align: center !important;
    color: #FFD700; 
}

.text {
    text-align: center;
}
"""

js = """
function createGradioAnimation() {
    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.fontSize = '2em';
    container.style.fontWeight = 'bold';
    container.style.textAlign = 'center';
    container.style.marginBottom = '20px';
    container.style.color = '#eba93f';

    var text = 'Welcome to your AI NutriCoach!';
    for (var i = 0; i < text.length; i++) {
        (function(i){
            setTimeout(function(){
                var letter = document.createElement('span');
                letter.style.opacity = '0';
                letter.style.transition = 'opacity 0.1s';
                letter.innerText = text[i];

                container.appendChild(letter);

                setTimeout(function() {
                    letter.style.opacity = '0.9';
                }, 50);
            }, i * 250);
        })(i);
    }

    var gradioContainer = document.querySelector('.gradio-container');
    gradioContainer.insertBefore(container, gradioContainer.firstChild);

    return 'Animation created';
}
"""
with gr.Blocks(theme=gr.themes.Citrus(), css=css, js=js) as demo:
    gr.Markdown("# How it works", elem_classes="title")
    gr.Markdown("Upload an image of your fridge content, enter your dietary restriction (if you have any!) and select a workflow type 'recipe' then click 'Analyze' to get recipe ideas.", elem_classes="text")
    gr.Markdown("Upload an image of a complete dish, leave dietary restriction blank and select a workflow type 'analysis' then click 'Analyze' to get nutritional insights.", elem_classes="text")
    gr.Markdown("You can also select one of the examples provided to autofill the input sections and click 'Analyze' right away!", elem_classes="text")

    with gr.Row():
        with gr.Column(scale=1, min_width=400):
            gr.Markdown("## Inputs", elem_classes="title")
            image_input = gr.Image(type="pil", label="Upload Image")
            dietary_input = gr.Textbox(label="Dietary Restrictions (optional)", placeholder="e.g., vegan")
            workflow_radio = gr.Radio(["recipe", "analysis"], label="Workflow Type")
            submit_btn = gr.Button("Analyze")
        
        with gr.Column(scale=2, min_width=600):
            # Place Examples directly under the Analyze button
            gr.Examples(
                examples=[
                    ["examples/food-1.jpg", "vegan", "recipe"],
                    ["examples/food-2.jpg", "", "analysis"],
                    ["examples/food-3.jpg", "keto", "recipe"],
                    ["examples/food-4.jpg", "", "analysis"],
                ],
                inputs=[image_input, dietary_input, workflow_radio],
                label="Try an Example: Select one of the examples below to autofil the input section then click Analyze"
            )
            gr.Markdown("## Results will appear here...", elem_classes="title")
            # result_display = gr.Markdown(height=800, )
            result_display = gr.Markdown(
                "<div style='border: 1px solid #ccc; "
                "padding: 1rem; text-align: center; "
                "color: #666;'>No results yet</div>",
                height=500
            )

    submit_btn.click(
        fn=analyze_food,
        inputs=[image_input, dietary_input, workflow_radio],
        outputs=result_display
    )

# Launch the Gradio interface
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=5000)

def detect_ingredients(image_path):
    """Detect ingredients from food image using vision model"""
    print("Loading vision model for ingredient detection...")
    vision_model = pipeline("image-classification", 
                           model="microsoft/resnet-50")
    
    image = Image.open(image_path)
    result = vision_model(image, top_k=10)
    
    ingredients = [item["label"].split(",")[0] for item in result]
    return ingredients

def filter_ingredients(ingredients, dietary_restrictions):
    """Filter ingredients based on dietary restrictions"""
    print(f"Filtering ingredients based on {dietary_restrictions} restrictions...")
    text_model = pipeline("text-generation", 
                         model="distilgpt2",
                         max_length=200)
    
    prompt = f"""
    Given the following ingredients: {', '.join(ingredients)}
    And these dietary restrictions: {dietary_restrictions}
    
    List only the ingredients that comply with these restrictions:
    """
    
    response = text_model(prompt)[0]["generated_text"]
    # Simple extraction of the generated ingredients
    filtered = response.split("restrictions:")[1].strip() if "restrictions:" in response else response
    return filtered

def generate_recipe(ingredients, dietary_restrictions):
    """Generate recipe suggestions based on ingredients and dietary preferences"""
    print("Generating recipe suggestions...")
    text_model = pipeline("text-generation", 
                         model="distilgpt2",
                         max_length=300)
    
    prompt = f"""
    Create a healthy recipe using some or all of these ingredients: {', '.join(ingredients)}
    The recipe should be suitable for people with these dietary restrictions: {dietary_restrictions}
    
    Recipe name:
    """
    
    response = text_model(prompt)[0]["generated_text"]
    return response

def analyze_nutrition(ingredients):
    """Analyze nutritional content of ingredients"""
    print("Analyzing nutritional content...")
    text_model = pipeline("text-generation", 
                         model="distilgpt2",
                         max_length=250)
    
    prompt = f"""
    Provide a nutritional analysis for a meal with these ingredients: {', '.join(ingredients)}
    
    Calories:
    Protein:
    Carbs:
    Fat:
    Vitamins:
    """
    
    response = text_model(prompt)[0]["generated_text"]
    return response

def handle_recipe_workflow(image_path, dietary_restrictions):
    """Run the recipe generation workflow"""
    print(f"\n=== Starting Recipe Workflow for {os.path.basename(image_path)} ===\n")
    
    # Step 1: Detect ingredients
    print("Step 1: Detecting ingredients...")
    ingredients = detect_ingredients(image_path)
    print(f"Detected ingredients: {', '.join(ingredients)}\n")
    
    # Step 2: Filter based on dietary restrictions
    print(f"Step 2: Filtering for {dietary_restrictions} diet...")
    filtered_result = filter_ingredients(ingredients, dietary_restrictions)
    print(f"Filtered ingredients: {filtered_result}\n")
    
    # Step 3: Generate recipe
    print("Step 3: Generating recipe suggestion...")
    recipe = generate_recipe(ingredients, dietary_restrictions)
    print("\n--- Recipe Suggestion ---\n")
    print(recipe)
    
    return recipe

def handle_analysis_workflow(image_path):
    """Run the food analysis workflow"""
    print(f"\n=== Starting Analysis Workflow for {os.path.basename(image_path)} ===\n")
    
    # Step 1: Detect ingredients
    print("Step 1: Detecting ingredients...")
    ingredients = detect_ingredients(image_path)
    print(f"Detected ingredients: {', '.join(ingredients)}\n")
    
    # Step 2: Analyze nutrition
    print("Step 2: Analyzing nutritional content...")
    nutrition = analyze_nutrition(ingredients)
    print("\n--- Nutritional Analysis ---\n")
    print(nutrition)
    
    return nutrition

def main():
    parser = argparse.ArgumentParser(description="AI NutriCoach - Food Analysis & Recipe Suggestions")
    parser.add_argument("image_path", help="Path to the food image")
    parser.add_argument("workflow", choices=["recipe", "analysis"], 
                        help="Workflow type: 'recipe' or 'analysis'")
    parser.add_argument("dietary_restrictions", nargs="?", default="none",
                        help="Dietary restrictions (for recipe workflow only)")
    
    args = parser.parse_args()
    
    # Validate image path
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found")
        return 1
    
    try:
        if args.workflow == "recipe":
            handle_recipe_workflow(args.image_path, args.dietary_restrictions)
        else:  # analysis
            handle_analysis_workflow(args.image_path)
            
        print("\nProcess completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())