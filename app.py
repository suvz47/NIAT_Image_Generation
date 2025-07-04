from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm import generate_engineered_prompt

app = FastAPI(title="Prompt Engineering API")

class PromptRequest(BaseModel):
    user_prompt: str

class PromptResponse(BaseModel):
    improved_prompt: str

@app.post("/engineer", response_model=PromptResponse)
async def engineer_prompt(request: PromptRequest):
    try:
        improved = generate_engineered_prompt(request.user_prompt)
        return PromptResponse(improved_prompt=improved)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))