from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel, Field
import uvicorn
from datetime import datetime
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List, Dict
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
import base64
import json


load_dotenv()

app = FastAPI(title="NilaMitra GenAI Microservice", version="1.0")

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

# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=model)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))


KB_DIR = "knowledge_base"
VECTOR_DB_DIR = "./chroma_db"

if not Path(VECTOR_DB_DIR).exists():
    print("Creating vector database from knowledge files...")
    docs = []
    for filename in os.listdir(KB_DIR):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(KB_DIR, filename))
            docs.extend(loader.load())
    
    for filename in os.listdir(KB_DIR):
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(KB_DIR, filename))
            docs.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    print("Vector database created successfully.")
else:
    print("Loading existing vector database...")
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings
    )

class Activity(BaseModel):
    date: str
    activity: str

class Profile(BaseModel):
    location: str
    land_size: str
    crop: str
    soil_type: str
    irrigation: str

class ExternalData(BaseModel):
    weather: str
    pest_alert: str

class QueryRequest(BaseModel):
    farmer_id: str
    query_text: str
    profile: Profile
    activities: List[Activity]
    external_data: ExternalData

class ImageQueryRequest(BaseModel):
    farmer_id: str
    profile: Profile
    activities: List[Activity]
    external_data: ExternalData
    query_text: str = ""

# Output Model
class QueryResponse(BaseModel):
    advisory_text_ml: str = Field(description="The full advisory text in Malayalam.")
    advisory_text_en: str = Field(description="A concise English summary of the advisory.")
    confidence: float = Field(description="The confidence score of the advisory from 0.0 to 1.0.")
    recommendations: List[str] = Field(description="A list of specific, actionable recommendations for the farmer.")
    metadata: Dict[str, str] = Field(description="Additional metadata, e.g., timestamp and model version.")

parser = PydanticOutputParser(pydantic_object = QueryResponse)

format_instructions = parser.get_format_instructions()

retriever = vectorstore.as_retriever()

rag_chain = (
    {
        "context": (lambda x: x["input"]) | retriever, # Pass the "input" string to the retriever
        "input": RunnablePassthrough(), # Pass the whole input dict to the next step
        "format_instructions": RunnablePassthrough() # Pass the parser instructions to the next step
    }
    | ChatPromptTemplate.from_messages([
        ("system", "You are a helpful farmer assistant named Krishi Sakhi. Your responses must be structured exactly as per the following instructions and schema. Use the provided context to answer the user's question.\n\n{format_instructions}\n\nContext:\n{context}"),
        ("human", "{input}"),
    ])
    | llm
    | parser
)

@app.post("/query", response_model=QueryResponse)
async def ask_advisor(request: QueryRequest):
    profile_json = request.profile.model_dump_json()
    external_data_json = request.external_data.model_dump_json()
    
    activities_str = ", ".join([f"'{a.activity}' on {a.date}" for a in request.activities])

    full_query = f"""
        User Question: {request.query_text}
        User Profile: {profile_json}
        Recent Activities: {activities_str}
        External Data: {external_data_json}
    """

    try:
        response = rag_chain.invoke({
            "input": full_query,
            "format_instructions": format_instructions
        })
        
        return response.model_dump()
    except Exception as e:
        print(f"Error during structured output generation: {e}")
        return QueryResponse(
            advisory_text_ml="ക്ഷമിക്കണം, ഒരു പിശക് സംഭവിച്ചു. (Sorry, an error occurred.)",
            advisory_text_en="An error occurred while generating the advisory.",
            confidence=0.0,
            recommendations=[],
            metadata={"timestamp": datetime.utcnow().isoformat(), "error": str(e)}
        ).model_dump()

@app.post("/image-query", reponse_model = QueryResponse)
async def advisory_with_image(
    image: UploadFile = Form(...),
    farmer_id: str = Form(...),
    profile: str = Form(...),
    activities: str = Form(...),
    external_data: str = Form(...),
    query_text: str = Form(None)
):
    #loading the image
    image_bytes = await image.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    # Re-build our complex input from the Form data (JSON strings)
    request_data = QueryRequest(
        farmer_id = farmer_id,
        query_text = query_text or "What is in this image?",
        profile = json.loads(profile),
        activities = json.loads(activities),
        external_data = json.loads(external_data)
    )

    full_query = f"""
        User Question: {request_data.query_text}
        User Profile: {request_data.profile.model_dump_json()}
        Recent Activities: {', '.join([f'{a.activity} on {a.date}' for a in request_data.activities])}
        External Data: {request_data.external_data.model_dump_json()}"""

    multimodal_input = [
        HumanMessage(
            content = [
                {'type':"text", "text": f"""Analyze this image based on the following context. {full_query_text}"""},
                {'type':"image_url", "image_url": {'url': f"data:image/{image.content_type};base64,{image_base64}"}}
            ]
        )
    ]

    llm = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash', google_api_key = os.getenv('GOOGLE_API_KEY'))

    try:
        response = llm.invoke(multimodal_input)
        return response.model_dump()
    except Exception as e:
        print(f"Error during structured output generation: {e}")
        return QueryResponse(
            advisory_text_ml="ക്ഷമിക്കണം, ഒരു പിശക് സംഭവിച്ചു. (Sorry, an error occurred.)",
            advisory_text_en="An error occurred while generating the advisory.",
            confidence=0.0,
            recommendations=[],
            metadata={"timestamp": datetime.utcnow().isoformat(), "error": str(e)}
        ).model_dump()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)