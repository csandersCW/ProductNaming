import os
import logging
from flask import Flask, request, jsonify, render_template
from openai import OpenAI, APIError

# --- Configuration ---
app = Flask(__name__, template_folder='templates') 
logging.basicConfig(level=logging.INFO)

# --- OpenAI Client Initialization ---
client = None
try:
    # Check for the user's preferred environment variable OPENAI_KEY
    openai_key = os.getenv("OPENAI_KEY")
    
    if openai_key:
        # Explicitly initialize with the key provided by OPENAI_KEY
        client = OpenAI(api_key=openai_key)
        logging.info("OpenAI client initialized using OPENAI_KEY.")
    else:
        # Fallback: Client will automatically check for the standard OPENAI_API_KEY
        client = OpenAI() 
        logging.info("OpenAI client initialized using default environment variable check.")

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
    """Handles the POST request and generates content from the OpenAI API."""
    
    # Pre-checks
    if not client:
        return jsonify({"error": "AI service is not configured. OPENAI_KEY or OPENAI_API_KEY is missing."}), 500
    
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
        
        # 3. Return the result to the user
        return jsonify({"text": generated_text})

    except APIError as e:
        logging.error(f"OpenAI API Error: {e}")
        return jsonify({"error": f"OpenAI API failed: {e.status_code}. Please check your key or rate limits."}), 500
    except Exception as e:
        logging.error(f"Internal Server Error: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

# --- Deployment Entry Point ---
if __name__ == '__main__':
    # When running locally for testing
    app.run(host='0.0.0.0', port=5000, debug=True)
else:
    # Required for production WSGI servers (like Gunicorn) to find the application instance.
    pass