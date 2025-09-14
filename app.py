import os
import streamlit as st
from dotenv import load_dotenv

from src.preprocessing import process_pdf, perform_ocr_on_images, combine_text_and_ocr
from src.retrieval import (
    build_text_vector_store,
    retrieve_text_context,
    build_image_vector_store,
    retrieve_image_context,
)
from src.graph import build_mdoc_agent_graph

# Load API keys
load_dotenv()

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
OUTPUT_FOLDER = "./output"
TEXT_SOURCES_DIR = os.path.join(OUTPUT_FOLDER, "combined_text")
IMAGE_SOURCES_DIR = os.path.join(OUTPUT_FOLDER, "images")
VECTORSTORE_DIR = "./vectorstore"
TEXT_PERSIST_DIR = os.path.join(VECTORSTORE_DIR, "chroma_db")
IMAGE_INDEX_PATH = os.path.join(VECTORSTORE_DIR, "faiss_index.bin")
IMAGE_METADATA_PATH = os.path.join(VECTORSTORE_DIR, "faiss_metadata.json")

os.makedirs("documents", exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# ------------------------------------------------------------------
# PIPELINE
# ------------------------------------------------------------------
def run_mdoc_agent(pdf_path: str, question: str):
    # Stage 1: Preprocessing
    if not os.path.exists(TEXT_SOURCES_DIR):
        text_files, image_files = process_pdf(pdf_path, OUTPUT_FOLDER)
        ocr_results = perform_ocr_on_images(image_files)
        combine_text_and_ocr(text_files, ocr_results, OUTPUT_FOLDER)

    # Stage 2: Build vector stores
    if not os.path.exists(TEXT_PERSIST_DIR):
        build_text_vector_store(TEXT_SOURCES_DIR, TEXT_PERSIST_DIR)
    if not os.path.exists(IMAGE_INDEX_PATH):
        build_image_vector_store(IMAGE_SOURCES_DIR, IMAGE_INDEX_PATH, IMAGE_METADATA_PATH)

    # Stage 2b: Retrieve
    retrieved_texts = retrieve_text_context(question, TEXT_PERSIST_DIR, k=4)
    retrieved_images = retrieve_image_context(question, IMAGE_INDEX_PATH, IMAGE_METADATA_PATH, k=4)

    # Stage 3–5: Agents
    app = build_mdoc_agent_graph()
    initial_state = {
        "question": question,
        "retrieved_texts": retrieved_texts,
        "retrieved_images": retrieved_images,
    }

    final_state = None
    for s in app.stream(initial_state):
        for node_name, node_output in s.items():
            final_state = s

    # Final Answer
    if final_state:
        if "summarizing_agent" in final_state:
            return final_state["summarizing_agent"].get(
                "final_answer", final_state["summarizing_agent"]
            )
        elif "summarizer_agent" in final_state:
            return final_state["summarizer_agent"].get(
                "final_answer", final_state["summarizer_agent"]
            )
    return "⚠️ No final state produced."


# ------------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------------
st.set_page_config(page_title="MDocAgent", layout="wide")
st.title("📄 MDocAgent – Multi-Modal Multi-Agent Document Understanding")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

# Upload document
uploaded_pdf = st.file_uploader("Upload a PDF document", type=["pdf"])
if uploaded_pdf:
    pdf_path = os.path.join("documents", uploaded_pdf.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.read())
    st.session_state.pdf_path = pdf_path
    st.success(f"📄 Loaded {uploaded_pdf.name}")

# Show chat history
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Agent:** {msg}")

# Ask question
question = st.text_input("Ask a question about the document:")
if st.session_state.pdf_path and question:
    if st.button("Send"):
        # Save user question
        st.session_state.chat_history.append(("user", question))

        with st.spinner("Thinking... ⏳"):
            answer = run_mdoc_agent(st.session_state.pdf_path, question)

        # Save agent answer
        st.session_state.chat_history.append(("agent", answer))
        st.rerun()  # refresh UI with new history
