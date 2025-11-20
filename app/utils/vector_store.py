from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Optional
import os


class VectorStoreManager:
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # IMPROVED: Better embedding model (optional upgrade)
        # You can also use: "sentence-transformers/all-mpnet-base-v2" for better quality
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}  # NEW: Normalize for better similarity
        )
        self.vector_store: Optional[FAISS] = None

    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create a new vector store from documents."""
        try:
            # NEW: Validate documents before creating vector store
            if not documents:
                raise ValueError("Cannot create vector store with empty document list")

            # NEW: Filter out empty documents
            valid_docs = [doc for doc in documents if doc.page_content.strip()]

            if not valid_docs:
                raise ValueError("No valid documents with content found")

            self.vector_store = FAISS.from_documents(
                documents=valid_docs,
                embedding=self.embeddings
            )
            return self.vector_store

        except Exception as e:
            raise Exception(f"Error creating vector store: {str(e)}")

    def add_documents(self, documents: List[Document]):
        """Add new documents to existing vector store."""
        if self.vector_store is None:
            self.create_vector_store(documents)
        else:
            # Filter empty documents
            valid_docs = [doc for doc in documents if doc.page_content.strip()]
            if valid_docs:
                self.vector_store.add_documents(valid_docs)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents."""
        if self.vector_store is None:
            raise Exception("Vector store not initialized")
        return self.vector_store.similarity_search(query, k=k)

    def save_local(self, path: str):
        """Save vector store locally."""
        if self.vector_store:
            self.vector_store.save_local(path)

    def load_local(self, path: str):
        """Load vector store from local storage."""
        if os.path.exists(path):
            self.vector_store = FAISS.load_local(
                path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
