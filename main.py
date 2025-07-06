from flask import Flask, request, jsonify
from gradio_client import Client, handle_file
import os
import requests
import uuid
from dotenv import load_dotenv # To load environment variables
from supabase import create_client, Client as SupabaseClient

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# --- Configuration for Temporary Image Storage ---
UPLOAD_FOLDER = 'temp_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Supabase Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET_NAME = os.getenv("SUPABASE_BUCKET_NAME", "virtual-try-extracted")

# --- Debugging: Print loaded environment variables (first few chars for security) ---
print(f"DEBUG: SUPABASE_URL (first 15 chars): {SUPABASE_URL[:15] if SUPABASE_URL else 'Not set'}")
print(f"DEBUG: SUPABASE_KEY (first 10 chars): {SUPABASE_KEY[:10] if SUPABASE_KEY else 'Not set'}")
print(f"DEBUG: SUPABASE_BUCKET_NAME: {SUPABASE_BUCKET_NAME}")


supabase: SupabaseClient = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Supabase client initialized successfully.")
    except Exception as e:
        print(f"Error initializing Supabase client: {e}")
        supabase = None
else:
    print("Supabase URL or Key not found or empty in environment variables. Results will NOT be uploaded to Supabase.")
    print("Please ensure SUPABASE_URL and SUPABASE_KEY are correctly set in your .env file and the file is in the same directory.")


def download_image(url, filename):
    """Downloads an image from a URL and saves it locally."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {url} to {filepath}")
        return filepath
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image from {url}: {e}")
        return None

def upload_to_supabase(file_path, destination_folder="tryon_results"):
    """
    Uploads a local file to Supabase Storage and returns its public URL.
    """
    if not supabase:
        raise Exception("Supabase client not initialized. Cannot upload files. Check backend logs for details.")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found for upload: {file_path}")

    filename = os.path.basename(file_path)
    unique_filename = f"{uuid.uuid4()}_{filename}" # Ensure unique file names
    storage_path = f"{destination_folder}/{unique_filename}"

    print(f"Attempting to upload {file_path} to Supabase at {SUPABASE_BUCKET_NAME}/{storage_path}...")

    try:
        # Determine content type based on file extension
        # Gradio often outputs .webp or .png
        content_type = 'image/webp' # Default
        if filename.lower().endswith('.png'):
            content_type = 'image/png'
        elif filename.lower().endswith('.jpeg') or filename.lower().endswith('.jpg'):
            content_type = 'image/jpeg'
        elif filename.lower().endswith('.gif'): # Add other common image types
            content_type = 'image/gif'


        with open(file_path, 'rb') as f:
            # The upload method typically returns a dict with 'path' and 'id' on success.
            # It does NOT return a response object with status_code.
            upload_result = supabase.storage.from_(SUPABASE_BUCKET_NAME).upload(storage_path, f.read(), {'content-type': content_type})
            
            # If the upload call completes without raising an exception, it was successful.
            # Now, retrieve the public URL.
            public_url = supabase.storage.from_(SUPABASE_BUCKET_NAME).get_public_url(storage_path)
            print(f"Uploaded to Supabase successfully, public URL: {public_url}")
            return public_url

    except Exception as e:
        print(f"Error during Supabase upload: {e}")
        # Re-raise the exception so the Flask endpoint catches it and returns a 500.
        raise

@app.route('/virtual-try-on', methods=['POST'])
def virtual_try_on():
    local_human_path = None
    local_garment_path = None
    local_gradio_output_path = None
    local_gradio_masked_path = None

    try:
        data = request.get_json()
        human_image_url = data.get('human_image_url')
        garment_image_url = data.get('garment_image_url')
        garment_description = data.get('garment_description', '')

        if not human_image_url or not garment_image_url:
            return jsonify({"error": "Missing human_image_url or garment_image_url"}), 400

        # 1. Download input images locally from the provided URLs for Gradio client
        human_image_filename = f"human_input_{uuid.uuid4()}_{os.path.basename(human_image_url).split('?')[0]}"
        garment_image_filename = f"garment_input_{uuid.uuid4()}_{os.path.basename(garment_image_url).split('?')[0]}"

        local_human_path = download_image(human_image_url, human_image_filename)
        local_garment_path = download_image(garment_image_url, garment_image_filename)

        if not local_human_path or not local_garment_path:
            raise Exception("Failed to download one or more input images from provided URLs.")

        # 2. Initialize the Gradio client and make the prediction
        client = Client("jallenjia/Change-Clothes-AI")
        print(f"Gradio Client initialized. Loaded as API: {client.src}")

        input_dict = {
            "background": handle_file(local_human_path),
            "layers": [],
            "composite": None
        }

        print("Calling Gradio API /tryon...")
        result = client.predict(
            dict=input_dict,
            garm_img=handle_file(local_garment_path),
            garment_des=garment_description,
            is_checked=True,
            is_checked_crop=False,
            denoise_steps=30,
            seed=-1,
            category="upper_body",
            api_name="/tryon"
        )
        print("Gradio API call complete.")
        print(f"Raw Gradio result: {result}")

        if isinstance(result, tuple) and len(result) == 2:
            local_gradio_output_path = result[0]
            local_gradio_masked_path = result[1]

            # 3. Upload these local Gradio output files to Supabase Storage
            processed_image_public_url = upload_to_supabase(local_gradio_output_path, "processed_images")
            masked_image_public_url = upload_to_supabase(local_gradio_masked_path, "masked_images")

            print(f"Returning processed_image_public_url: {processed_image_public_url}")
            print(f"Returning masked_image_public_url: {masked_image_public_url}")

            return jsonify({
                "output_url": processed_image_public_url,
                "masked_url": masked_image_public_url
            }), 200
        else:
            return jsonify({"error": "Unexpected API response format from Gradio API. Expected a tuple of 2 local file paths."}), 500

    except Exception as e:
        print(f"Error in virtual try-on endpoint: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up all temporary local files (inputs and outputs)
        if local_human_path and os.path.exists(local_human_path):
            print(f"Removing temporary file: {local_human_path}")
            os.remove(local_human_path)
        if local_garment_path and os.path.exists(local_garment_path):
            print(f"Removing temporary file: {local_garment_path}")
            os.remove(local_garment_path)
        if local_gradio_output_path and os.path.exists(local_gradio_output_path):
            print(f"Removing temporary file: {local_gradio_output_path}")
            os.remove(local_gradio_output_path)
        if local_gradio_masked_path and os.path.exists(local_gradio_masked_path):
            print(f"Removing temporary file: {local_gradio_masked_path}")
            os.remove(local_gradio_masked_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)