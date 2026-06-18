import os
import time
import chromadb
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Initialize API and Storage Clients
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
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=text
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"Embedding error: {e}")
        return []

def train_system_on_directory(source_folder="./medical_knowledge_base"):
    if not os.path.exists(source_folder):
        os.makedirs(source_folder)
        print(f"📁 Created an empty folder named '{source_folder}'.")
        print("👉 Drop your medical text files (.txt) into that folder and re-run this script!")
        return

    # 2. Configure a smart text chunker
    # Splits massive books/articles into overlapping paragraphs so no context gets lost
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       # The size of each data block
        chunk_overlap=100     # Keeps continuity between paragraphs
    )

    files = [f for f in os.listdir(source_folder) if f.endswith('.txt')]
    if not files:
        print(f"📭 No text documents found inside '{source_folder}' folder.")
        return

    print(f"📚 Found {len(files)} documents to train on. Starting processing pipeline...")

    for file_name in files:
        file_path = os.path.join(source_folder, file_name)
        print(f"📖 Processing: {file_name}...")
        
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Break the large document into smart paragraphs
        chunks = text_splitter.split_text(raw_text)
        print(f"✂️ Split {file_name} into {len(chunks)} contextual fragments.")

        for idx, chunk in enumerate(chunks):
            # Clean up spacing issues
            clean_chunk = chunk.replace("\n", " ").strip()
            if not clean_chunk:
                continue

            # Generate vectors
            vector = get_embedding(clean_chunk)
            if vector:
                # Store unique IDs safely based on filename and index tracking
                unique_id = f"{file_name}_chunk_{idx}_{int(time.time())}"
                
                collection.add(
                    embeddings=[vector],
                    documents=[clean_chunk],
                    metadatas=[{"source": file_name}],
                    ids=[unique_id]
                )
                print(f"   ✅ Archived fragment {idx+1}/{len(chunks)} into vector database disk.")
                time.sleep(0.5) # Prevent rate limits

    print(f"\n🎉 Training complete! Database now contains {collection.count()} total clinical records.")

if __name__ == "__main__":
    train_system_on_directory()