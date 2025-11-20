import sys
import os
from typing import Dict, Any, List
from groq import Groq
import json
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.vector_store import VectorStoreManager
from app.utils.document_processor import DocumentProcessor
from app.config.settings import settings


class RAGModel:
    def __init__(self):
        self.document_processor = DocumentProcessor(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

        # NEW: Store multiple documents for comparison
        self.documents = {}  # {doc_id: {filename, stats, vector_store, chunks, total_text}}

        self.vector_store_manager = VectorStoreManager(
            embedding_model=settings.EMBEDDING_MODEL
        )

        # Groq client - simple and reliable
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)

        self.chat_histories: Dict[str, List[Dict[str, str]]] = {}
        self.document_stats = {}

    def get_session_history(self, session_id: str) -> List[Dict[str, str]]:
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = []
        return self.chat_histories[session_id]

    def process_document(self, file_path: str) -> str:
        """Process single document (existing functionality)"""
        try:
            chunks = self.document_processor.load_pdf(file_path)

            if not chunks:
                raise ValueError("No text extracted from PDF")

            chunks = [c for c in chunks if c.page_content.strip()]
            if not chunks:
                raise ValueError("No readable text after processing")

            # Calculate stats
            total_text = " ".join([c.page_content for c in chunks])
            self.document_stats = {
                "filename": os.path.basename(file_path),
                "total_words": len(total_text.split()),
                "total_characters": len(total_text),
                "total_chunks": len(chunks),
            }

            # Create vector store
            self.vector_store_manager.create_vector_store(chunks)

            return (f"✅ Processed '{self.document_stats['filename']}'\n"
                    f"📊 {self.document_stats['total_chunks']} chunks | "
                    f"{self.document_stats['total_words']:,} words")

        except Exception as e:
            return f"❌ Error: {str(e)}"

    # ==================== NEW: COMPARISON METHODS ====================

    def process_multiple_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process multiple documents for comparison with progress tracking"""
        results = {}
        total_files = len(file_paths)

        for idx, file_path in enumerate(file_paths, 1):
            doc_id = os.path.basename(file_path).replace('.pdf', '').replace(' ', '_')

            print(f"[{idx}/{total_files}] Processing: {os.path.basename(file_path)}")

            try:
                # Process each document
                chunks = self.document_processor.load_pdf(file_path)

                if not chunks:
                    results[doc_id] = {'status': 'error', 'error': 'No text extracted'}
                    continue

                chunks = [c for c in chunks if c.page_content.strip()]

                if not chunks:
                    results[doc_id] = {'status': 'error', 'error': 'No readable text'}
                    continue

                print(f"[{idx}/{total_files}] Creating embeddings for {len(chunks)} chunks...")

                # Calculate stats
                total_text = " ".join([c.page_content for c in chunks])

                # Create separate vector store for this document
                doc_vector_store = VectorStoreManager(
                    embedding_model=settings.EMBEDDING_MODEL
                )
                doc_vector_store.create_vector_store(chunks)

                print(f"[{idx}/{total_files}] ✅ Completed: {os.path.basename(file_path)}")

                # Store document info
                self.documents[doc_id] = {
                    'filename': os.path.basename(file_path),
                    'chunks': chunks,
                    'total_text': total_text,
                    'stats': {
                        'total_words': len(total_text.split()),
                        'total_characters': len(total_text),
                        'total_chunks': len(chunks)
                    },
                    'vector_store': doc_vector_store
                }

                results[doc_id] = {
                    'status': 'success',
                    'filename': os.path.basename(file_path),
                    'stats': self.documents[doc_id]['stats']
                }

            except Exception as e:
                print(f"[{idx}/{total_files}] ❌ Error: {str(e)}")
                results[doc_id] = {
                    'status': 'error',
                    'error': str(e)
                }

        return results

    def compare_documents(self, doc_ids: List[str], comparison_aspects: List[str] = None) -> Dict[str, Any]:
        """Compare multiple documents on various aspects"""

        if not comparison_aspects:
            comparison_aspects = [
                "skills and technologies",
                "work experience and roles",
                "education and certifications",
                "key achievements",
                "overall strengths",
                "potential areas for growth"
            ]

        comparison_results = {}

        for aspect in comparison_aspects:
            comparison_results[aspect] = {}

            for doc_id in doc_ids:
                if doc_id not in self.documents:
                    comparison_results[aspect][doc_id] = "Document not found"
                    continue

                try:
                    # Get relevant information for this aspect
                    doc_info = self.documents[doc_id]
                    vector_store = doc_info['vector_store']

                    # Search for relevant content
                    relevant_docs = vector_store.similarity_search(
                        f"information about {aspect}",
                        k=3
                    )
                    context = "\n".join([doc.page_content for doc in relevant_docs])

                    # Use Groq to extract structured information
                    prompt = f"""Analyze this document section and extract information about {aspect}.
Be specific and concise. List key points as bullet points.

Document section:
{context[:1500]}

Extract information about {aspect}:"""

                    messages = [
                        {"role": "system",
                         "content": "You are a document analyzer. Extract specific information concisely in bullet points."},
                        {"role": "user", "content": prompt}
                    ]

                    response = self.groq_client.chat.completions.create(
                        model=settings.LLM_MODEL,
                        messages=messages,
                        temperature=0.1,
                        max_tokens=400,
                    )

                    comparison_results[aspect][doc_id] = response.choices[0].message.content.strip()

                except Exception as e:
                    comparison_results[aspect][doc_id] = f"Error: {str(e)}"

        return comparison_results

    def get_recommendation(self, doc_ids: List[str], job_role: str = "") -> str:
        """Get AI recommendation for which candidate/document is best"""

        # Gather all comparison data
        comparison_data = self.compare_documents(doc_ids)

        # Build comprehensive context
        context = "Document Comparison Analysis:\n\n"
        for doc_id in doc_ids:
            if doc_id not in self.documents:
                continue
            context += f"\n### Document: {self.documents[doc_id]['filename']}\n"
            for aspect, candidates in comparison_data.items():
                if doc_id in candidates:
                    context += f"**{aspect}:**\n{candidates[doc_id]}\n\n"

        # Get recommendation
        role_context = f" for the role of '{job_role}'" if job_role else ""

        prompt = f"""Based on the document comparison below, provide a comprehensive recommendation{role_context}.

{context}

Provide your analysis in this format:

## Overall Recommendation
[Which document/candidate is the strongest and why - be specific]

## Individual Strengths
[List key strengths of each candidate]

## Best Fit Analysis
[Explain which candidate is best suited and why]

## Key Differentiators
[What sets the top candidate apart]

Be specific, actionable, and professional."""

        messages = [
            {"role": "system", "content": "You are an expert analyst providing detailed, actionable recommendations."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.groq_client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=1200,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error generating recommendation: {str(e)}"

    def extract_structured_data(self, doc_id: str) -> Dict[str, Any]:
        """Extract structured data from a document"""

        if doc_id not in self.documents:
            return {"error": "Document not found"}

        doc_info = self.documents[doc_id]
        text = doc_info['total_text'][:4000]

        prompt = f"""Extract structured information from this document.
    Provide a JSON response with these fields (use "N/A" if not found):
    {{
      "name": "Full name",
      "email": "Email address",
      "phone": "Phone number",
      "skills": ["skill1", "skill2", "skill3"],
      "experience_years": 0,
      "education": ["degree1", "degree2"],
      "certifications": ["cert1", "cert2"],
      "key_achievements": ["achievement1", "achievement2", "achievement3"]
    }}

    Document text:
    {text}

    Respond with ONLY valid JSON, no other text."""

        messages = [
            {"role": "system", "content": "You are a data extraction specialist. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.groq_client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=800,
            )

            result = response.choices.message.content.strip()

            # Clean markdown code blocks
            triple_backtick = "```"

            if result.startswith(triple_backtick):
                # Remove opening backticks
                parts = result.split(triple_backtick, 1)
                if len(parts) > 1:
                    result = parts[1]

                # Remove 'json' label if present
                if result.startswith('json'):
                    result = result[4:].strip()

                # Remove closing backticks
                if triple_backtick in result:
                    result = result.split(triple_backtick)[0].strip()

            # Parse and return JSON
            return json.loads(result)

        except json.JSONDecodeError:
            return {
                "error": "Could not parse structured data",
                "raw_response": result[:200] if 'result' in locals() else "No response"
            }
        except Exception as e:
            return {"error": f"Extraction failed: {str(e)}"}

    def clear_comparison_data(self):
        """Clear all comparison documents"""
        self.documents = {}
        return "Comparison data cleared"

    # ==================== END COMPARISON METHODS ====================

    def ask_question(self, question: str, session_id: str = "default") -> Dict[str, Any]:
        """Ask question (existing functionality)"""
        if not self.vector_store_manager.vector_store:
            return {
                "answer": "Please upload and process a document first.",
                "sources": [],
                "session_id": session_id
            }

        try:
            # Handle statistics questions
            ql = question.lower().strip()
            if any(kw in ql for kw in ["how many words", "word count", "total words", "page count", "document size"]):
                if self.document_stats:
                    s = self.document_stats
                    answer = (
                        f"**Document Statistics:**\n\n"
                        f"📄 File: {s['filename']}\n"
                        f"📊 Total Words: {s['total_words']:,}\n"
                        f"🔤 Characters: {s['total_characters']:,}\n"
                        f"📑 Chunks: {s['total_chunks']}"
                    )

                    history = self.get_session_history(session_id)
                    history.append({"role": "user", "content": question})
                    history.append({"role": "assistant", "content": answer})

                    return {"answer": answer, "sources": ["Calculated from document"], "session_id": session_id}

            # Retrieve relevant chunks
            relevant_docs = self.vector_store_manager.similarity_search(question, k=6)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            # Build messages for Groq
            system_prompt = """You are a document assistant. Answer using ONLY the provided context.

Rules:
- Extract facts, names, dates, numbers from context
- If asked for summary, cover main points
- If asked for details, be specific
- If not in context, say: "I cannot find that information"
- Do not add information not in context"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"}
            ]

            # Call Groq API
            response = self.groq_client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
            )

            answer = response.choices[0].message.content.strip()

            # Clean answer
            for prefix in ["Answer:", "Response:", "A:"]:
                if answer.startswith(prefix):
                    answer = answer[len(prefix):].strip()
                    break

            # Get sources
            sources = [f"{doc.page_content[:200]}..." for doc in relevant_docs[:3]]

            # Update history
            history = self.get_session_history(session_id)
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})

            return {"answer": answer, "sources": sources, "session_id": session_id}

        except Exception as e:
            return {"answer": f"Error: {str(e)}", "sources": [], "session_id": session_id}

    def get_chat_history(self, session_id: str = "default") -> List[Dict[str, str]]:
        return self.get_session_history(session_id)

    def clear_chat_history(self, session_id: str = "default") -> str:
        if session_id in self.chat_histories:
            self.chat_histories[session_id] = []
            return f"Chat history cleared for session: {session_id}"
        return f"No chat history found for session: {session_id}"

    def list_active_sessions(self) -> List[str]:
        return list(self.chat_histories.keys())
