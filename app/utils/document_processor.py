import sys
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # NEW: Better separators
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """Load and split PDF document into chunks."""
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"PDF file not found: {file_path}")

            loader = PyPDFLoader(file_path)
            pages = loader.load()

            # NEW: Check if any text was extracted
            if not pages:
                raise ValueError("PDF loaded but contains no pages")

            # NEW: Check if pages have content
            total_text = "".join([page.page_content for page in pages])
            if not total_text.strip():
                raise ValueError("PDF contains no extractable text. It may be an image-based PDF requiring OCR.")

            # Split documents into chunks
            chunks = self.text_splitter.split_documents(pages)

            # NEW: Filter empty chunks
            chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

            if not chunks:
                raise ValueError("Document splitting produced no valid chunks")

            return chunks

        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

    def process_text(self, text: str) -> List[Document]:
        """Process raw text into chunks."""
        if not text.strip():
            raise ValueError("Cannot process empty text")

        documents = [Document(page_content=text)]
        chunks = self.text_splitter.split_documents(documents)
        return chunks
