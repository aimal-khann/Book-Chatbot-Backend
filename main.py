import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from openai import OpenAI
from dotenv import load_dotenv

# Setup
logging.basicConfig(level=logging.INFO)
load_dotenv()

# Config
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "humanoid_ai_book"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"

# Initialize App & Clients
app = FastAPI()
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# CORS (Critical for Frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    query: str 

@app.get("/")
def home():
    return {"status": "online", "message": "RAG Brain is Running 🧠"}

@app.post("/ask")
def ask_question(request: AskRequest):
    # Handle "query" or "message" to be safe
    user_query = request.query
    logging.info(f"Received query: {user_query}")

    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # 1. Embed Query
        emb_res = openai_client.embeddings.create(
            input=[user_query],
            model=EMBEDDING_MODEL
        )
        query_vector = emb_res.data[0].embedding

        # 2. Search Qdrant
        search_results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=5
        ).points

        # 3. Build Context
        context_text = ""
        sources = []
        
        if not search_results:
            return {
                "reply": "I couldn't find any relevant information in the textbook.",
                "sources": []
            }

        for hit in search_results:
            text = hit.payload.get("text", "")
            source = hit.payload.get("source", "Unknown")
            context_text += f"---\nSource: {source}\n{text}\n"
            if source not in [s.get('title') for s in sources]:
                sources.append({"title": source})

        # 4. Generate Answer
        system_prompt = (
            "You are an AI Tutor for a Robotics Textbook. "
            "Answer the student's question using ONLY the context below. "
            "If the answer isn't in the context, say you don't know."
        )
        
        completion = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_query}"}
            ]
        )
        
        answer = completion.choices[0].message.content
        return {"reply": answer, "sources": sources}

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
