from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

llm = ChatOllama(model="tinyllama")
retriever = db.as_retriever(search_kwargs={"k": 4})

print("=" * 40)
print("  XenonStruct AI - Construction Assistant")
print("=" * 40)
print("Type your question below. Type 'quit' to exit.\n")

while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""You are XenonStruct AI, a construction safety assistant.
Use the context below to answer the question.

Context:
{context}

Question: {question}
Answer:"""
    response = llm.invoke(prompt)
    print(f"\nXenonStruct AI: {response.content}\n")