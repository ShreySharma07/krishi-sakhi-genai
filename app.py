from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
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

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=model)

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
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
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

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful farmer assistant named Krishi Sakhi. Your responses must be in Malayalam based on the following context:\n\n{context}"),
    ("human", "{input}")
])

retriever = vectorstore.as_retriever()

document_chain = create_stuff_documents_chain(llm, prompt_template)
retrieval_chain = create_retrieval_chain(retriever, document_chain)




@app.post("/api/ask")
async def ask_advisor(query: Query):
    print(f"Received contextual query from user {query.user_id}: {query.text}")

    try:
        response = retrieval_chain.invoke({
            'question':query.text,
            'farmer_profile':query.farmer_profile
        })
        return {'response':response.content}
    except Exception as e:
        print(f"LLM call failed: {e}")
        return {"response": "ക്ഷമിക്കണം, എനിക്ക് ഇപ്പോൾ നിങ്ങളെ സഹായിക്കാൻ കഴിയില്ല. (Sorry, I can't help you right now.)"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)