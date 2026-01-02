import os
import time
import uuid
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from google import genai
import PyPDF2

# ==============================
# CONFIG & INITIALIZATION
# ==============================
# We set template_folder to '.' if your index.html is in the same directory
# Otherwise, create a folder named 'templates' and put index.html inside it.
app = Flask(__name__, template_folder='templates')
CORS(app)  # Allow frontend access

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# API Key is provided by the execution environment
API_KEY = "" 
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_ID = "gemini-2.5-flash"

# ==============================
# PDF TEXT EXTRACTION
# ==============================
def extract_text_from_pdf(pdf_path):
    """Extracts raw text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print(f"PDF Extraction Error: {e}")
    return text.strip()

# ==============================
# CORE ATS INTELLIGENCE (SINGLE PASS)
# ==============================
def run_ats_engine(resume_text, jd_text):
    """
    Calls Gemini with exponential backoff to handle rate limits.
    """
    prompt = f"""
    You are an elite Career Coach and ATS Expert. 
    Analyze the provided resume against the job description.
    
    TASK 1: Provide Analysis (Match Score, Strengths, Gaps).
    
    
    RESUME:
    {resume_text}

    JOB DESCRIPTION:
    {jd_text}

    RETURN THE RESULT STRICTLY IN THIS FORMAT:

    ATS Match: <number>%

    Strengths:
    - [Bullet points]

    Improvement Suggestions:
    - [Bullet points]

    
    Rules:
    - Output must be text-only.
    - Optimized resume must look professional and be ready to use.
    
    """

    # Exponential Backoff Strategy for API Stability
    delays = [1, 2, 4, 8, 16]
    for delay in delays:
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt
            )
            return response.text.strip()
        except Exception:
            # If rate limited or transient error, wait and retry
            time.sleep(delay)
            
    return "Error: AI engine is currently overloaded. Please try again in 30 seconds."

# ==============================
# ROUTES
# ==============================

@app.route("/", methods=["GET"])
def index():
    """Renders the ResumeIQ 2.0 Frontend."""
    # Ensure index.html is located in a folder named 'templates'
    return render_template("main.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    """Main API endpoint for resume analysis."""
    # 1. Input Validation
    if "resume" not in request.files:
        return jsonify({"error": "Resume file missing"}), 400

    resume_file = request.files["resume"]
    jd_text = request.form.get("job_description", "").strip()

    if not jd_text:
        return jsonify({"error": "Job description required"}), 400

    try:
        # 2. Save PDF safely with unique ID
        filename = f"{uuid.uuid4()}.pdf"
        pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        resume_file.save(pdf_path)

        # 3. Extract text
        resume_text = extract_text_from_pdf(pdf_path)

        if not resume_text:
            if os.path.exists(pdf_path): os.remove(pdf_path)
            return jsonify({"error": "Could not extract text from the uploaded PDF"}), 400

        # 4. Run AI Analysis via Gemini
        ats_result = run_ats_engine(resume_text, jd_text)

        # 5. Cleanup temp file to maintain privacy and disk space
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        # 6. Response exactly as the frontend's renderResults() expects
        return jsonify({
            "parsed_resume": resume_text[:3000], 
            "parsed_job_description": jd_text[:3000],
            "ats_result": ats_result
        })

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
    # Standard Flask port. The frontend fetch('http://localhost:8080/analyze') matches this.
    app.run(port=8080, debug=True)