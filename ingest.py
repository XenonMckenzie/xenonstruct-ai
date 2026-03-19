import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Load all PDFs from the docs folder
loader = PyPDFDirectoryLoader("docs/")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

# Embed using Ollama (free, runs locally)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = FAISS.from_documents(chunks, embeddings)
db.save_local("faiss_index")

print(f"Done! {len(chunks)} chunks indexed from {len(documents)} pages.")