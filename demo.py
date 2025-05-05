import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("WATSONX_API_KEY")
URL = os.getenv("WATSONX_URL")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")

if not API_KEY:
    raise Exception("WATSONX_API_KEY is missing or empty in the environment variables.")

# Step 1: Get IAM Token
def get_iam_token(api_key):
    iam_url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key,
    }
    response = requests.post(iam_url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception(f"Failed to get IAM token: {response.text}")

# Step 2: Make a Request to WatsonX
def make_request(token, url, project_id):
    endpoint = f"{url}/v1/projects/{project_id}/some-endpoint"  # Replace 'some-endpoint' with the actual API endpoint
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        # Add your request payload here
    }
    response = requests.get(endpoint, headers=headers, json=payload)  # Use POST if required
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Request failed: {response.text}")

# Main Execution
try:
    iam_token = get_iam_token(API_KEY)
    response = make_request(iam_token, URL, PROJECT_ID)
    print("Response:", response)
except Exception as e:
    print("Error:", e)