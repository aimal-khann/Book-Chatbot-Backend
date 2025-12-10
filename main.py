import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Added for CORS
from pydantic import BaseModel
from qdrant_client import QdrantClient
from openai import OpenAI
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# Configuration from environment variables
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-4o"
COLLECTION_NAME = "humanoid_ai_book"
TOP_K_CHUNKS = 3

# Initialize clients
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    reply: str
    sources: list[dict] # { "file": "path", "start_line": N, "end_line": M, "title": "Title" }

def generate_embedding(text: str) -> list[float]:
    """Generates OpenAI embedding for a single text."""
    response = openai_client.embeddings.create(input=[text], model=OPENAI_EMBEDDING_MODEL)
    return response.data[0].embedding

def get_answer_from_gpt(query: str, context: str) -> str:
    """Generates an answer using GPT-4o based on the provided context."""
    prompt = (
        "You are an AI Tutor for a 'Physical AI & Humanoid Robotics' textbook. "
        "The user will provide a Query, which might be a specific question OR a highlighted text fragment.\n"
        "1. If the Query is a question, answer it using the Context below.\n"
        "2. If the Query is a text fragment, explain that concept using the Context below.\n"
        "3. STRICTLY restrict your answer to the provided Context. If the context doesn't support an answer, "
        "politely state that you don't have enough information.\n\n"
        f"Context:\n{context}\n\nQuery: {query}\nAnswer:"
    )
    
    response = openai_client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7,
        stream=False
    )
    return response.choices[0].message.content

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Receives a user query, searches Qdrant for relevant context,
    and generates an answer using OpenAI's GPT-4o.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # 1. Embed user query
        query_embedding = generate_embedding(request.query)

        # 2. Search Qdrant using query_points (Matching original text file)
        search_result = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=TOP_K_CHUNKS,
        ).points

        context_chunks = []
        sources = []
        for hit in search_result:
            if hit.payload:
                context_chunks.append(hit.payload.get("text", ""))
                sources.append({
                    "file": hit.payload.get("filename"),
                    "start_line": hit.payload.get("start_line"), # These might need to be extracted from metadata
                    "end_line": hit.payload.get("end_line"),     # or from the ingester if not directly in payload
                    "title": hit.payload.get("source") # Using source as a placeholder for title for now
                })
        
        full_context = "\n\n".join(context_chunks)

        if not full_context:
            reply = "I'm sorry, but I couldn't find relevant information in the documents to answer your question."
            sources = [] # No sources if no context
        else:
            # 3 & 4. Construct prompt and generate answer using GPT-4o
            reply = get_answer_from_gpt(request.query, full_context)
        
        return AskResponse(reply=reply, sources=sources)

    except Exception as e:
        logging.error(f"Error processing ask_question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error occurred.")

@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running!"}
