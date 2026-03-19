import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

st.set_page_config(
    page_title="XenonStruct AI",
    page_icon="🏗️",
    layout="centered"
)

st.markdown("""
    <h1 style='text-align: center; color: #F4A300;'>🏗️ XenonStruct AI</h1>
    <p style='text-align: center; color: gray;'>Construction Safety Assistant — Powered by OSHA Knowledge</p>
    <hr>
""", unsafe_allow_html=True)

@st.cache_resource
def load_chain():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )
    retriever = db.as_retriever(search_kwargs={"k": 4})
    return llm, retriever

llm, retriever = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask XenonStruct AI a construction question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("XenonStruct AI is thinking..."):
            docs = retriever.invoke(prompt)
            context = "\n\n".join([d.page_content for d in docs])
            full_prompt = f"""You are XenonStruct AI, a construction safety assistant.
Use the context below to answer the question.

Context:
{context}

Question: {prompt}
Answer:"""
            response = llm.invoke([HumanMessage(content=full_prompt)])
            answer = response.content
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})