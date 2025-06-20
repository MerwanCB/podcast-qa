import os
import streamlit as st
from dotenv import load_dotenv

# --- Correct, modern imports ---
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# Load environment variables
load_dotenv()

# --- App Configuration ---
# Add custom CSS for background and style
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(120deg, #f8fafc 0%, #e0e7ff 100%);
    }
    .main .block-container {
        background: rgba(255,255,255,0.95);
        border-radius: 18px;
        padding: 2rem 2rem 2rem 2rem;
        box-shadow: 0 4px 24px 0 rgba(0,0,0,0.07);
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1.5px solid #6366f1;
        background: #f1f5f9;
    }
    .stTextInput>div>div>input:focus {
        border: 1.5px solid #6366f1 !important;
        box-shadow: 0 0 0 2px #c7d2fe;
    }
    .stButton>button {
        background: linear-gradient(90deg, #6366f1 0%, #818cf8 100%);
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        color: white !important;
        background: linear-gradient(90deg, #6366f1 0%, #6366f1 100%);
        border: none;
    }
    .fancy-box {
        background: #eef2ff;
        border-left: 5px solid #6366f1;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add a sidebar with info and links
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/727/727240.png", width=80)
st.sidebar.title("About Podcast Q&A")
st.sidebar.markdown(
    """
    **Podcast Q&A** lets you ask questions to your favorite podcast transcripts using Google's Gemini.
    
    - [GitHub Repo](https://github.com/MerwanCB)
    - [Lex Fridman Podcast](https://lexfridman.com/podcast/)
    """
)
st.sidebar.markdown("---")

st.set_page_config(page_title="Podcast Q&A", layout="centered")
st.title("🎙️ Ask Questions to Your Podcasts")
st.caption(
    "This app uses Google's Gemini to answer questions based on podcast transcripts."
)


# --- Helper Functions ---
@st.cache_resource(show_spinner="Loading models and index...")
def get_index():
    """Loads the vector store index from the storage directory."""
    if not os.path.exists("./storage"):
        st.error("Index not found! Please run engine.py first to create the index.")
        st.stop()

    # --- THIS IS THE FIX ---
    # Configure the models globally BEFORE loading the index.
    # This ensures LlamaIndex uses the correct models for all subsequent operations.
    Settings.llm = GoogleGenAI(model_name="models/gemini-1.5-flash-latest")
    Settings.embed_model = GoogleGenAIEmbedding(model_name="models/embedding-001")

    # Load the existing index using the global settings.
    # No need to pass models here anymore.
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
    return index


def main():
    """Main function to run the Streamlit app."""
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY not found. Please set it in your .env file.")
        st.stop()

    # Load the index (which now correctly sets the models)
    index = get_index()
    query_engine = index.as_query_engine(streaming=True, similarity_top_k=3)

    # Initialize chat history and input in session state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "show_sources" not in st.session_state:
        st.session_state["show_sources"] = {}
    if "question_input" not in st.session_state:
        st.session_state["question_input"] = ""

    def send_question():
        user_question = st.session_state["question_input"].strip()
        if user_question:
            with st.spinner("Thinking..."):
                response = query_engine.query(user_question)
                answer = "".join([chunk for chunk in response.response_gen])
                sources = [
                    {
                        "file": node.metadata.get("file_name", "N/A"),
                        "score": node.score,
                        "text": node.text,
                    }
                    for node in response.source_nodes
                ]
                st.session_state["chat_history"].append(
                    {
                        "question": user_question,
                        "answer": answer,
                        "sources": sources,
                    }
                )
                st.session_state["show_sources"][
                    len(st.session_state["chat_history"]) - 1
                ] = False
            st.session_state["question_input"] = ""

    # Use columns for layout
    def toggle_sources(idx):
        st.session_state["show_sources"][idx] = not st.session_state[
            "show_sources"
        ].get(idx, False)

    col1, col2 = st.columns([2, 1])
    with col1:
        # Display chat history first (top-down)
        for idx, chat in enumerate(st.session_state["chat_history"]):
            st.markdown(
                f'<div class="fancy-box"><b>Q:</b> {chat["question"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="fancy-box" style="background:#e0e7ff;"><b>A:</b> {chat["answer"]}</div>',
                unsafe_allow_html=True,
            )
            st.button(
                (
                    "Show Sources"
                    if not st.session_state["show_sources"].get(idx, False)
                    else "Hide Sources"
                ),
                key=f"show_sources_{idx}",
                on_click=toggle_sources,
                args=(idx,),
            )
            if st.session_state["show_sources"].get(idx, False):
                for src in chat["sources"]:
                    st.markdown(f"**Source File:** `{src['file']}`")
                    st.markdown(f"**Score:** `{src['score']:.2f}`")
                    st.markdown("---")
                    st.markdown(src["text"])
        # Input field at the bottom
        st.markdown(
            '<div class="fancy-box">🤖 <b>Ask a question about your podcast transcript below!</b></div>',
            unsafe_allow_html=True,
        )
        st.text_input(
            "Ask your question:",
            key="question_input",
            on_change=send_question,
        )
    with col2:
        st.image(
            "https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExMzB5cGN6djd5dm4yMzhnaGU1eXo3eWRncXIyNmVnendmemlzb2VsZyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/hXPHgJXpg1TnQncuHW/giphy.gif",
            caption="I'm here to help you with your podcast questions!",
            use_container_width=True,
        )


# --- Main execution ---
if __name__ == "__main__":
    main()
