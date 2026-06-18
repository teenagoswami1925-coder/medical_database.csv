import streamlit as st
import pandas as pd
from google import genai

# =====================================================================
# 1. INITIALIZE & CONFIGURATION
# =====================================================================
import os

# This checks if the terminal environment variable exists. 
# If it doesn't find one, it safely falls back to your manual string.
API_KEY = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)

# Load your local CSV dataset
# Ensure your CSV file is in the exact same folder as this script!
@st.cache_data
def load_medical_data():
    try:
        # Replace 'medical_dataset.csv' with your actual filename
        df = pd.read_csv("medical_dataset.csv")
        # Standardize columns to lowercase to prevent mistakes
        df.columns = df.columns.str.lower()
        return df
    except FileNotFoundError:
        st.error("⚠️ 'medical_dataset.csv' not found. Please put your file in this folder.")
        return None

df = load_medical_data()

# =====================================================================
# 2. LOCAL KNOWLEDGE SEARCH ENGINE (RAG)
# =====================================================================
def find_medical_context(user_query, dataframe):
    """Searches your CSV for matching terms using column positions instead of names."""
    if dataframe is None or dataframe.empty:
        return ""
    
    query_words = user_query.lower().split()
    
    # Dynamically grab whatever the first two columns are named
    query_column = dataframe.columns[0]   # This maps to your first column
    response_column = dataframe.columns[1] # This maps to your second column
    
    # Search for rows where the first column contains our keywords
    match_condition = dataframe[query_column].astype(str).str.lower().apply(
        lambda x: any(word in x for word in query_words)
    )
    matches = dataframe[match_condition]
    
    if not matches.empty:
        # Take the matched clinical answer from the second column
        best_match = matches.iloc[0][response_column]
        return f"\n[Verified Clinical Guideline]: {best_match}\n"
    return ""

# =====================================================================
# 3. STREAMLIT WEB INTERFACE
# =====================================================================
st.set_page_config(page_title="Clinical AI Assistant", page_icon="🩺")
st.title("🩺 Medical Assistant Chatbot")
st.caption("Powered by Gemini 2.5 & Your Local Clinical Datasets")

# Hardcoded medical system boundary instruction
SYSTEM_INSTRUCTION = (
    "You are an expert medical assistant.\n\n"
    "STRICT FORMAT RULES:\n"
    "- Respond ONLY in bullet points, each starting with *\n"
    "- Begin with one short introductory sentence\n"
    "- End with: '* Consult a doctor for proper diagnosis and treatment.'\n"
    "- Do NOT use paragraphs, numbered lists, or any other format\n"
    "- If a [Verified Clinical Guideline] is provided, base your answer exclusively on it\n"
    "- If no guideline is provided, state that no data is available \u2014 do not guess"
)

# Initialize persistent chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat logs
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Capture live user inputs
if user_input := st.chat_input("Describe symptoms or ask a medical question..."):
    
    # 1. Display user query instantly
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Extract facts out of your CSV file matching this query
    local_facts = find_medical_context(user_input, df)
    
    # 3. Package the query along with your verified CSV data 
    if local_facts:
        structured_prompt = (
            f"{local_facts}\nUser Question: {user_input}\n\n"
            f"Answer based ONLY on the verified clinical guideline above."
        )
    else:
        structured_prompt = (
            f"{user_input}\n\n"
            f"IMPORTANT: No verified clinical data is available for this query. "
            f"Respond with: 'No verified clinical data available. Please consult a doctor.' "
            f"Do NOT make up or guess medical information."
        )

    # 4. Stream response live from Gemini
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Stream generation creates an interactive, typing-like experience
        response_stream = client.models.generate_content_stream(
            model='gemini-2.5-flash',
            contents=structured_prompt,
            config={
                "system_instruction": SYSTEM_INSTRUCTION,
                "temperature": 0.1
            }
        )
        
        for chunk in response_stream:
            full_response += chunk.text
            response_placeholder.markdown(full_response + "▌")
            
        response_placeholder.markdown(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})
