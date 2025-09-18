from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


app = FastAPI()

class Query(BaseModel):
    user_id: int
    text:str
    language: str

@app.post("/api/ask")
async def ask_advisor(query: Query):

    print(f"Received query from user {query.user_id}: {query.text}")
    
    # Return a static response for the MVP
    return {"response": "നമസ്കാരം! നിങ്ങളുടെ ചോദ്യം എനിക്ക് ലഭിച്ചു. (Hello! I have received your question.)"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)