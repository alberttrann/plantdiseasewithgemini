import google.generativeai as genai
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv
import os

# Load api key trong file .env vào 
load_dotenv()

# Config GenerativeAI API key bằng environment variable đã load vào 
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up model config để gen text
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

# safety settings
safety_settings = [
    {"category": f"HARM_CATEGORY_{category}", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
    for category in ["HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]
]

# Khởi tạo GenerativeModel với model name, configuration với safety settings
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
    safety_settings=safety_settings,
)
# Function để đọc image data từ file path
def read_image_data(file_path):
    image_path = Path(file_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Could not find image: {image_path}")
    return {"mime_type": "image/jpeg", "data": image_path.read_bytes()}

# Function để tạo response dựa trên prompt với image path
def generate_gemini_response(prompt, image_path):
    image_data = read_image_data(image_path)
    response = model.generate_content([prompt, image_data])
    return response.text

# System prompt cho model
input_prompt = """
You are a world‐leading plant pathologist. Your mission is to analyze a photographic sample of a plant, diagnose any diseases or disorders, and provide actionable management advice. Respond in five clearly labeled sections.

1. Disease Identification  
   • Name the disease(s) or disorder(s), with scientific and common names.  
   • If multiple issues are present, list each separately.

2. Observable Symptoms  
   • Describe the visible signs on leaves, stems, roots, flowers, or fruit.  
   • Note severity (mild, moderate, severe) and distribution (localized, systemic).

3. Likely Causes & Risk Factors  
   • Explain probable pathogens (fungi, bacteria, viruses, nematodes) or abiotic stress (nutrient deficiency, drought, chemical damage).  
   • Mention environmental or cultural conditions that may have contributed.

4. Management & Treatment Recommendations  
   • Give both immediate treatments (e.g., fungicides, pruning, biocontrols) and long‐term strategies (crop rotation, resistant varieties).  
   • Include dosage/rate guidelines, timing, and safety precautions.

5. Preventive Measures & Monitoring  
   • Suggest ongoing practices to prevent recurrence (irrigation management, sanitation, soil health).  
   • Propose a simple scouting schedule and key indicators to watch.

**Important:**  
- If the image is too low‐resolution or ambiguous, request additional photos or details.  
- Conclude with an estimated confidence level (High/Medium/Low) based on image quality and symptom clarity.  
- Keep each section concise (2–4 sentences).
"""

# Function để process file được upload lên và tạo response 
def process_uploaded_files(files):
    file_path = files[0].name if files else None
    response = generate_gemini_response(input_prompt, file_path) if file_path else None
    return file_path, response

# Set up giao diện Gradio
with gr.Blocks() as demo:
    file_output = gr.Textbox()
    image_output = gr.Image()
    combined_output = [image_output, file_output]

    # Upload button cho images
    upload_button = gr.UploadButton(
        "Click to Upload an Image",
        file_types=["image"],
        file_count="multiple",
    )
     # Set up upload button để trigger processing function
    upload_button.upload(process_uploaded_files, upload_button, combined_output)

# Launch Gradio UI với debug mode 
demo.launch(debug=True)
