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

# --- 🚀 UPGRADE: Read 10 chunks instead of 5 ---
TOP_K_CHUNKS = 10 

# Initialize App & Clients
app = FastAPI()

# Robust Client Initialization
try:
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    logging.error(f"Failed to initialize clients: {e}")

# CORS (Allows your website to talk to this brain)
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
    # Support both "query" and "message" inputs
    user_query = request.query or request.message
    
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
            limit=TOP_K_CHUNKS
        ).points

        # 3. Build Context
        context_text = ""
        sources = []
        
        for hit in search_results:
            text = hit.payload.get("text", "")
            source = hit.payload.get("source", "Unknown")
            # Only add unique sources
            if source not in [s.get('title') for s in sources]:
                sources.append({"title": source})
            context_text += f"---\nSource: {source}\n{text}\n"

        # 4. Generate Answer (Smarter, Politer Prompt)
        system_prompt = (
            "You are an expert AI Tutor for a Robotics Textbook. "
            "Your goal is to be helpful, friendly, and accurate.\n\n"
            "RULES:\n"
            "1. If the user says 'Hi', 'Hello', or 'Thanks', reply politely and offer help with the book.\n"
            "2. Answer questions STRICTLY based on the provided Context below.\n"
            "3. If the answer is not in the Context, say: 'I'm sorry, that topic isn't covered in the textbook yet.'\n"
            "4. Do not make up information outside the context."
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
        # Return a polite error instead of crashing the frontend
        return {
            "reply": "I'm having trouble connecting to my brain right now. Please try again in a moment.",
            "sources": []
        }
