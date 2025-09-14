# src/retrieval.py

import os
import json
from typing import List

import chromadb
# from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import torch
import open_clip
import faiss
import numpy as np
from PIL import Image


# -------------------------------------------------------------------------
# TEXT VECTOR STORE
# -------------------------------------------------------------------------

def build_text_vector_store(text_sources_dir: str, persist_directory: str):
    """
    Builds a ChromaDB vector store from text files in a directory.
    """
    documents = []
    for filename in os.listdir(text_sources_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(text_sources_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract page number from filename safely
            try:
                # e.g. "doc_name_page_1.txt" → "1"
                page_str = filename.split("_")[-1].split(".")[0]
                page_number = int(page_str)
            except Exception:
                page_number = -1  # fallback if parsing fails

            documents.append(
                Document(
                    page_content=content,
                    metadata={"source": filename, "page": page_number},
                )
            )

    # 1. Chunking
    text_splitter = SemanticChunker(OpenAIEmbeddings())
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} semantic chunks.")

    # 2. Embedding
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # 3. Indexing (Vector Store Creation)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=persist_directory,
    )
    vector_store.persist()
    print(f"Text vector store created and persisted at {persist_directory}")


def retrieve_text_context(query: str, persist_directory: str, k: int = 4) -> List[Document]:
    """
    Retrieves relevant text chunks from the ChromaDB vector store.
    """
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(query)
    return retrieved_docs


# -------------------------------------------------------------------------
# IMAGE VECTOR STORE
# -------------------------------------------------------------------------

# Global setup for CLIP model (load once)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
clip_model.to(device)
clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")


def build_image_vector_store(image_sources_dir: str, index_path: str, metadata_path: str):
    """
    Builds a FAISS index for images in a directory using CLIP embeddings.
    """
    image_paths = [
        os.path.join(image_sources_dir, f)
        for f in os.listdir(image_sources_dir)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    image_paths.sort()  # Ensure consistent order

    all_embeddings = []
    print(f"Generating CLIP embeddings for {len(image_paths)} images...")
    with torch.no_grad():
        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            image_tensor = clip_preprocess(image).unsqueeze(0).to(device)
            embedding = clip_model.encode_image(image_tensor)
            embedding /= embedding.norm(dim=-1, keepdim=True)  # Normalize
            all_embeddings.append(embedding.cpu().numpy())

    embeddings_matrix = np.vstack(all_embeddings)

    # FIX: use dimension, not shape
    embedding_dim = embeddings_matrix.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_matrix)

    faiss.write_index(index, index_path)
    with open(metadata_path, "w") as f:
        json.dump(image_paths, f)

    print(f"Image vector store (FAISS index) created and saved at {index_path}")


def retrieve_image_context(query: str, index_path: str, metadata_path: str, k: int = 4) -> List[str]:
    """
    Retrieves relevant image paths from the FAISS index based on a text query.
    """
    index = faiss.read_index(index_path)
    with open(metadata_path, "r") as f:
        image_paths = json.load(f)

    # Embed the text query
    with torch.no_grad():
        text_tensor = clip_tokenizer([query]).to(device)
        query_embedding = clip_model.encode_text(text_tensor)
        query_embedding /= query_embedding.norm(dim=-1, keepdim=True)
        query_embedding = query_embedding.cpu().numpy()

    # Search the FAISS index
    distances, indices = index.search(query_embedding, k)

    # Flatten indices in case FAISS returns nested lists
    indices = indices.flatten().tolist()
    retrieved_image_paths = [image_paths[i] for i in indices if i < len(image_paths)]

    return retrieved_image_paths
