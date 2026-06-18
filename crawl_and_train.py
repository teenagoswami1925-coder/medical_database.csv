import os
import time
import re
import requests
from bs4 import BeautifulSoup
import chromadb
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize clients
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
    try:
        response = client.models.embed_content(model="gemini-embedding-2", contents=text)
        return response.embeddings[0].values
    except Exception:
        return []

def scrape_and_train(url: str):
    print(f"\n🌐 Connecting to web target: {url}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"❌ Failed to reach website. Status code: {response.status_code}")
            return
            
        # Parse and strip HTML boilerplate (navbars, footers, ads)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Target main content containers common to wikis and blogs
        main_content = soup.find('main') or soup.find('article') or soup.find('div', id='content') or soup.body
        
        # Extract raw text and clean white spaces
        raw_text = main_content.get_text(separator=' ')
        clean_text = re.sub(r'\s+', ' ', raw_text).strip()
        
        print(f"📥 Successfully downloaded and extracted {len(clean_text)} characters of text context.")
        
        # Configure text chunker
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_text(clean_text)
        print(f"✂️ Slicing webpage data into {len(chunks)} optimized memory blocks...")

        # Ingest chunks into database disk
        source_name = url.split('/')[-1] or "web_source"
        for idx, chunk in enumerate(chunks):
            vector = get_embedding(chunk)
            if vector:
                unique_id = f"web_{source_name}_{idx}_{int(time.time())}"
                collection.add(
                    embeddings=[vector],
                    documents=[chunk],
                    metadatas=[{"source": url}],
                    ids=[unique_id]
                )
                print(f"   ✅ Indexed segment {idx+1}/{len(chunks)}")
                time.sleep(0.5) # Protect API limits
                
        print(f"\n🎉 Training Complete! Total database records now at: {collection.count()}")
        
    except Exception as e:
        print(f"❌ Scraping engine execution failed: {e}")

if __name__ == "__main__":
    # Example usage for testing
    target_link = input("🔗 Paste any medical page URL to train your chatbot: ")
    if target_link.strip():
        scrape_and_train(target_link.strip())