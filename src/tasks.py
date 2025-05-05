from src.crew import NutriCoachRecipeCrew, NutriCoachAnalysisCrew

# worker_tasks.py
def process_nutri_task(image_path, dietary_restrictions, workflow_type):
    """
    This function processes the image and generates either a recipe or analysis.
    """
    inputs = {
        'uploaded_image': image_path,
        'dietary_restrictions': dietary_restrictions,
        'workflow_type': workflow_type
    }

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
        return {"error": "Invalid workflow type."}

    crew_obj = crew_instance.crew()
    final_output = crew_obj.kickoff(inputs=inputs).to_dict()
    final_output["workflow_type"] = workflow_type
    return final_output