import os
import json
import logging
from flask import Flask, request, jsonify, render_template
from openai import OpenAI, APIError
import wandb 

# --- Configuration ---
# Set template_folder='templates' so Flask knows where to look for index.html
app = Flask(__name__, template_folder='templates') 
logging.basicConfig(level=logging.INFO)

# --- W&B Initialization ---
# W&B is initialized once at startup. It will look for WANDB_API_KEY environment variable.
try:
    if os.environ.get("WANDB_KEY"):
        # Initialize a run for tracking the entire application session or deployment
        wandb.init(project="northflank-product-generator", job_type="web_app_inference", anonymous="allow")
        logging.info("Weights & Biases initialized successfully.")
    else:
        logging.warning("WANDB_API_KEY not found. W&B logging is disabled. Please set the environment variable on Northflank.")
except Exception as e:
    logging.error(f"Failed to initialize W&B: {e}")

# --- OpenAI Client Initialization ---
try:
    # The client will automatically look for the OPENAI_API_KEY environment variable.
    client = OpenAI()
    logging.info("OpenAI client initialized.")
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {e}")
    client = None

# --- Constants ---
MODEL_NAME = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "You are a highly irreverent, slightly broken marketing algorithm specializing in generating product "
    "names, slogans, and descriptions that are utterly ridiculous, absurd, and nonsensical. "
    "Your names must sound professionally stupid, like something a startup would try to pitch. "
    "Respond only with the product details, formatted neatly using markdown headings."
)


# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """Serves the main HTML page from the templates folder."""
    return render_template('index.html')

@app.route('/', methods=['POST'])
def generate_name():
    """Handles the POST request, generates content, and logs the call to W&B."""
    
    # Pre-checks
    if not client:
        return jsonify({"error": "AI service is not configured. OPENAI_API_KEY is missing."}), 500
    
    # Check if W&B is active
    is_wandb_enabled = wandb.run is not None and wandb.run.mode != 'disabled'

    try:
        data = request.get_json()
        product_type = data.get('product_type', 'unnecessary gadget').strip()
        key_feature = data.get('key_feature', 'vibrates every time you think about squirrels').strip()

        if not product_type or not key_feature:
            return jsonify({"error": "Please provide both a product type and a key feature."}), 400

        prompt_user = (
            f"Generate a product concept for a {product_type} with the key feature: '{key_feature}'. "
            "Format the response strictly with these three sections: Name, Slogan, and Description."
        )
        
        # 1. Prepare messages for OpenAI
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_user}
        ]

        # 2. Call the OpenAI API
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.9,
            max_tokens=512
        )

        generated_text = response.choices[0].message.content
        
        # 3. Log the call to Weights & Biases (if enabled)
        if is_wandb_enabled:
            usage = response.usage
            
            # Create a W&B Table entry for structured logging
            table = wandb.Table(columns=["timestamp", "model", "prompt_tokens", "completion_tokens", "total_tokens", "input", "output"])
            
            table.add_data(
                wandb.Timestamp.now(),
                MODEL_NAME,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
                prompt_user,
                generated_text
            )
            
            # Log the table and usage metrics
            wandb.log({
                "llm_calls": table,
                "token_usage/prompt": usage.prompt_tokens,
                "token_usage/completion": usage.completion_tokens,
                "token_usage/total": usage.total_tokens,
                "product_type_input": product_type,
                "key_feature_input": key_feature
            })
            logging.info("W&B log entry created successfully.")

        # 4. Return the result to the user
        return jsonify({"text": generated_text})

    except APIError as e:
        logging.error(f"OpenAI API Error: {e}")
        # Log error to W&B if possible
        if is_wandb_enabled:
            wandb.log({"error": f"OpenAI API failed: {e.status_code}", "error_details": str(e)})
        return jsonify({"error": f"OpenAI API failed: {e.status_code}. Please check your key or rate limits."}), 500
    except Exception as e:
        logging.error(f"Internal Server Error: {e}")
        # Log error to W&B if possible
        if is_wandb_enabled:
            wandb.log({"error": "Internal Server Error", "error_details": str(e)})
        return jsonify({"error": "An internal server error occurred."}), 500

# --- Deployment Entry Point ---
if __name__ == '__main__':
    # When running locally for testing
    app.run(host='0.0.0.0', port=5000, debug=True)
else:
    # Required for production WSGI servers (like Gunicorn) to find the application instance.
    pass