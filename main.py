from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import time
import anyio  
import chromadb
from google import genai
from google.genai import types
from sse_starlette.sse import EventSourceResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is missing!")
client = genai.Client(api_key=API_KEY)

chroma_client = chromadb.PersistentClient(path="./medical_db")
collection = chroma_client.get_or_create_collection(
    name="clinical_guidelines", 
    metadata={"hnsw:space": "cosine"}
)

def get_embedding(text: str) -> list:
    for attempt in range(3):
        try:
            response = client.models.embed_content(model="gemini-embedding-2", contents=text)
            return response.embeddings[0].values
        except Exception:
            if attempt < 2: time.sleep(2)
    return []

def find_semantic_context(user_query, threshold=0.75):
    user_vector = get_embedding(user_query)
    if not user_vector or collection.count() == 0:
        return []
    results = collection.query(query_embeddings=[user_vector], n_results=3)
    if results and results['distances'] and len(results['distances'][0]) > 0:
        contexts = []
        for distance, doc in zip(results['distances'][0], results['documents'][0]):
            if (1.0 - distance) >= threshold:
                contexts.append(doc)
        return contexts
    return []

class ChatMessageSchema(BaseModel):
    content: str
    isUser: bool
    isVerified: bool = False

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessageSchema]

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_input = request.message
    
    try:
        local_fact = await anyio.to_thread.run_sync(
            find_semantic_context, user_input, 0.75
        )
    except Exception as embedding_error:
        print(f"Embedding rate limit hit: {embedding_error}")
        local_fact = []

    gemini_history = []
    for msg in request.history:
        if "Hello! Describe your symptoms" in msg.content: continue
        role = "user" if msg.isUser else "model"
        gemini_history.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.content)]))

    if local_fact:
        context_str = "\n".join(f"- {c}" for c in local_fact)
        prompt_content = (
            f"Verified Clinical References:\n{context_str}\n\n"
            f"User Query: {user_input}\n\n"
            f"Answer based ONLY on the verified references above."
        )
    else:
        prompt_content = (
            f"{user_input}\n\n"
            f"IMPORTANT: No verified clinical data is available for this query. "
            f"Respond with: 'No verified clinical data available. Please consult a doctor.' "
            f"Do NOT make up or guess medical information."
        )

    SYSTEM_INSTRUCTION = (
        "You are an expert medical assistant.\n\n"
        "STRICT FORMAT RULES:\n"
        "- Respond ONLY in bullet points, each starting with *\n"
        "- Begin with one short introductory sentence\n"
        "- End with: '* Consult a doctor for proper diagnosis and treatment.'\n"
        "- Do NOT use paragraphs, numbered lists, or any other format\n"
        "- If Verified References are provided, base your answer exclusively on them\n"
        "- If no references are provided, state that no data is available \u2014 do not guess"
    )

    async def event_generator():
        try:
            yield {"event": "metadata", "data": "verified" if local_fact else "unverified"}
            
            # This inner function handles the stream collection safely on the background thread
            def pull_stream_chunks():
                chat = client.chats.create(
                    model="gemini-2.5-flash",
                    history=gemini_history,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.1
                    )
                )
                response_stream = chat.send_message_stream(prompt_content)
                
                # Consume the synchronous chunks into a list safely within the thread 
                # to prevent StopIteration escaping to the async loop
                chunks = []
                for chunk in response_stream:
                    if chunk.text:
                        chunks.append(chunk.text)
                return chunks
            
            # Fetch all generated chunks in one thread-safe operation
            text_chunks = await anyio.to_thread.run_sync(pull_stream_chunks)
            
            # Yield them sequentially out to the SSE stream connection
            for text in text_chunks:
                yield {"event": "message", "data": text}
                    
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print("⚠️ CRITICAL: Gemini Free Tier Quota Exceeded!")
                yield {
                    "event": "message", 
                    "data": "⚠️ **Server Traffic Safeguard Active**\n\n* The medical server is receiving a high volume of requests right now.\n* Please wait exactly 20 seconds for the cloud quota pool to clear.\n* Hit the 'Send' button again shortly to complete your triage assessment."
                }
            else:
                yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_generator())