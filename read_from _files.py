import os
from pathlib import Path
from typing import List

from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from langchain.schema import LLMResult

import google.generativeai as genai

# ---- Configure Gemini API key ----
# genai.configure(api_key=os.environ.get("your_gemini_api_key"))

# ---- Helpers ----
def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def chunk_text_safe(text: str, chunk_size=50, overlap=10) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        end = min(len(words), i + chunk_size)
        chunk = " ".join(words[i:end])
        chunks.append(chunk)
        print(f"Chunk {len(chunks)}: {chunk}")
        # move forward safely
        i += chunk_size - overlap
        if i < 0:
            i = 0  # safety check
    return chunks


# ---- Gemini Embeddings for LangChain ----
class GeminiEmbeddings(Embeddings):
    def __init__(self, model: str = "models/embedding-001"):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for txt in texts:
            resp = genai.embed_content(model=self.model, content=txt)
            embeddings.append(resp["embedding"])
            print(f"Embedded: {txt[:30]}... -> {resp['embedding'][:5]}...")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# ---- Gemini LLM Wrapper using Base LLM ----
class GeminiLLM(LLM):
    model_name: str = "gemini-2.5-flash"

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(self, prompt: str, stop=None) -> str:
        model = genai.GenerativeModel(self.model_name)
        resp = model.generate_content(prompt)
        return resp.text

    def _agenerate(self, prompts, stop=None):
        results = [self._call(p, stop=stop) for p in prompts]
        return LLMResult(generations=[[{"text": r} for r in results]])


# ---- Read file ----
    file_path = Path(r"C:\Users\mayur pallakki\Desktop\github_mcp\docs\sample.txt")
text = read_txt(file_path)

# ---- Chunk the text ----
chunks = chunk_text_safe(text)

# ---- Add to Chroma vectorstore ----
embeddings = GeminiEmbeddings()
vectordb = Chroma(
    persist_directory="./chroma_gemini",
    collection_name="docs",
    embedding_function=embeddings
)
vectordb.add_texts(
    texts=chunks,
    metadatas=[{"source": str(file_path), "chunk_index": i} for i in range(len(chunks))]
)

print(f"✅ Ingested {len(chunks)} chunks from {file_path}")

# ---- RAG Query ----
def answer_question(question: str, top_k=5):
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=GeminiLLM(),
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain.run(question)

# ---- Run a test query ----
answer = answer_question("whats my dog's name?")
print("\n📌 Answer:", answer)
