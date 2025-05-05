import gradio as gr
import base64
import time
from src.crew import NutriCoachRecipeCrew, NutriCoachAnalysisCrew
from dotenv import load_dotenv
from src.tasks import process_nutri_task
from redis import Redis
from rq import Queue


# Setup Redis connection and RQ queue
redis_conn = Redis()
q = Queue(connection=redis_conn)

# Load environment variables from .env file
load_dotenv()

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
    Wrapper function for Gradio interface using message queue,
    but synchronously waits for the result.
    """
    
    # Save the uploaded image
    image_path = "./uploaded_image.jpg"
    image.save(image_path)

    # Enqueue the task to RQ queue
    job = q.enqueue(process_nutri_task, image_path, dietary_restrictions, workflow_type)

    # Wait for the job to finish and get the result (synchronous wait)
    while not job.is_finished:
        time.sleep(0.5)  # Sleep for 0.5 seconds to avoid tight looping

    result = job.result  # Get the result from the job

    if "error" in result:
        return f"❌ Error: {result['error']}"

    # Process and format the output based on the workflow type
    if workflow_type == "recipe":
        recipe_markdown = format_recipe_output(result)
        return recipe_markdown
    elif workflow_type == "analysis":
        nutrient_markdown = format_analysis_output(result)
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