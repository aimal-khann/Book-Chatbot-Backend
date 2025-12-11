import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from openai import OpenAI
from dotenv import load_dotenv

# Setup Logging
logging.basicConfig(level=logging.INFO)
load_dotenv()

# Config
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "humanoid_ai_book"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"

# --- 🚀 UPGRADE 1: READ MORE ---
TOP_K_CHUNKS = 10 

app = FastAPI()

try:
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    logging.error(f"Failed to initialize clients: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    query: str = None
    message: str = None 

@app.get("/")
def home():
    return {"status": "online", "message": "RAG Brain is Running 🧠"}

@app.post("/ask")
def ask_question(request: AskRequest):
    user_query = request.query or request.message
    
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # 1. Embed
        emb_res = openai_client.embeddings.create(
            input=[user_query],
            model=EMBEDDING_MODEL
        )
        query_vector = emb_res.data[0].embedding

        # 2. Search
        search_results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=TOP_K_CHUNKS
        ).points

        # 3. Context
        context_text = ""
        sources = []
        for hit in search_results:
            text = hit.payload.get("text", "")
            source = hit.payload.get("source", "Unknown")
            if source not in [s.get('title') for s in sources]:
                sources.append({"title": source})
            context_text += f"---\nSource: {source}\n{text}\n"

        # --- 🚀 UPGRADE 2: SMARTER PROMPT ---
        system_prompt = (
            "You are an expert AI Tutor for a Robotics Textbook. "
            "1. If the user query includes noise like 'Read More', IGNORE it and answer the core topic.\n"
            "2. If the user greets you (hi, hello), reply politely and ask about the book.\n"
            "3. Answer strictly based on the Context below.\n"
            "4. If the answer isn't in the Context, say: 'I'm sorry, I couldn't find that specific detail in the textbook.'"
        )
        
        completion = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_query}"}
            ]
        )
        
        return {"reply": completion.choices[0].message.content, "sources": sources}

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return {"reply": "I'm having trouble connecting right now. Please try again.", "sources": []}
