# AI NutriCoach - Building Nutrition Coach using Multi-Agent System (MAS) and multimodal AI

AI NutriCoach is an AI-powered nutrition assistant that utilizes advanced vision models and natural language processing to analyze food images. It detects ingredients, filters them based on dietary restrictions, estimates calories, provides a detailed nutrient breakdown, and generates personalized recipe suggestions. This project showcases the integration of **CrewAI**, **WatsonX**, and other AI tools to deliver insightful and customized nutritional feedback.

## Features

- **Ingredient Detection**  
  Automatically detects ingredients from user-uploaded food images using an AI-powered vision model.

- **Dietary Filtering**  
  Filters identified ingredients based on user-defined dietary restrictions (e.g., vegan, gluten-free, low-carb).

- **Calorie Estimation**  
  Provides an approximate calorie count based on detected ingredients.

- **Nutrient Analysis**  
  Offers a detailed breakdown of key nutrients, including proteins, carbohydrates, fats, vitamins, and minerals.

- **Health Evaluation**  
  Summarizes the overall healthiness of the meal and provides a comprehensive health score.

- **Recipe Suggestions**  
  Generates personalized recipe recommendations based on detected ingredients and dietary preferences.

## How It Works

The project is structured using the **CrewAI** framework, which manages AI agents and workflows to perform two primary tasks:

1. **Recipe Workflow**  
   - Detects ingredients from an image.
   - Filters them based on dietary restrictions.
   - Suggests relevant recipes.

2. **Analysis Workflow**  
   - Estimates calories from a food image.
   - Performs nutrient analysis.
   - Provides a health evaluation summary.

## Installation

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- Git

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shashidharbabu/ai-nutricoach.git
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   - Create a `.env` file in the root directory and add the following:
   ```ini
   WATSONX_API_KEY=your_watsonx_api_key
   WATSONX_URL=your_watsonx_url
   WATSONX_PROJECT_ID=your_watsonx_project_id
   ```

## Usage

### Running the Application

To analyze a food image or generate recipe suggestions, use the following commands:

#### 1. Generate Recipe Suggestions
```bash
python main.py <image_path> <dietary_restrictions> recipe
```
**Example:**
```bash
python main.py food.jpg vegan recipe
```

#### 2. Perform Food Analysis
```bash
python main.py <image_path> analysis
```
**Example:**
```bash
python main.py food.jpg analysis
```

#### 3. Training (Future Feature - TODO)
```bash
python main.py train <n_iterations> <output_filename> <image_path> <dietary_restrictions> <workflow_type>
```

## File Structure

```
ai-nutricoach/
│
├── config/
│   ├── agents.yaml               # Configuration for AI agents
│   └── tasks.yaml                # Configuration for tasks and workflows
│
├── src/
│   ├── crew.py                   # Crew definitions (agents, tasks, workflows)
│   ├── tools.py                   # Utility functions for ingredient detection, filtering, etc.
│   └── main.py                    # Main script to run the application
│
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## Credits

This project was inspired by **Hailey Q’s** work and her talk on **Multimodal AI** at the **IBM TechXchange Dev Event**.

For inquiries or support, please contact [Hailey Thao Quach](mailto:hailey@haileyq.com).

