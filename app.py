from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

app = FastAPI()

model = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    google_api_key = model,
    temperature=0.4
)

class Query(BaseModel):
    user_id: str
    text:str
    language: str
    farmer_profile: dict

@app.post("/api/ask")
async def ask_advisor(query: Query):
    print(f"Received contextual query from user {query.user_id}: {query.text}")
    
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful farmer assistant for farmers based in Kerala. Your response must be in Malayalam/English based on the user's preference."),
            ("human", "Here is the farmer's profile: {farmer_profile}"),
            ("human", "Farmer's question: {question}"),
        ]
    )

    chain = prompt_template | llm

    try:
        response = chain.invoke({
            'question':query.text,
            'farmer_profile':query.farmer_profile
        })
        return {'response':response.content}
    except Exception as e:
        print(f"LLM call failed: {e}")
        return {"response": "ക്ഷമിക്കണം, എനിക്ക് ഇപ്പോൾ നിങ്ങളെ സഹായിക്കാൻ കഴിയില്ല. (Sorry, I can't help you right now.)"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)