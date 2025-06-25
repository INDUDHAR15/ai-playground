from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import google.generativeai as genai
from pydantic import BaseModel
import os

# Configure the Gemini API (ensure the API key is set as an environment variable)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable must be set")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Define the request body
class ChatRequest(BaseModel):
    message: str

class PromptInput(BaseModel):
    input_prompt: str

# Create FastAPI app instance
app = FastAPI()

# Mount the static directory to serve static files (like HTML)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Jinja2 templates for rendering HTML
templates = Jinja2Templates(directory="templates")

# Route for the home page, rendering the index.html file
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Define the chat endpoint
@app.post("/prompt-gemini")
async def generate_content(data: PromptInput):
    try:
        response = model.generate_content(data.input_prompt)
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {e}")
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)