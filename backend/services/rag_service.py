from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
import os

from core.config import settings

class RAGService:
    def __init__(self):
        self.embeddings = GPT4AllEmbeddings()
        self.vectorstore = self._initialize_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", kwargs={"k": 5})

    def _initialize_vectorstore(self):
        if os.path.exists(settings.EMBEDDINGS_DIR) and os.listdir(settings.EMBEDDINGS_DIR):
            print("Loading existing Chroma vectorstore...")
            return Chroma(
                embedding_function=self.embeddings,
                persist_directory=str(settings.EMBEDDINGS_DIR)
            )
        else:
            print("Creating new Chroma vectorstore from PDFs...")
            docs = []
            files = list(settings.POLICIES_DIR.glob("*.pdf"))
            for file in files:
                loader = PyPDFLoader(str(file))
                docs.extend(loader.load())
            
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=str(settings.EMBEDDINGS_DIR)
            )
            print("Embeddings Completed.")
            return vectorstore

rag_service = RAGService()
