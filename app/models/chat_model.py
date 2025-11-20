import sys
import os
from typing import Dict, Any, List

# Path setup for absolute imports
current_file = os.path.abspath(__file__)
models_dir = os.path.dirname(current_file)  # models directory
app_dir = os.path.dirname(models_dir)  # app directory
project_root = os.path.dirname(app_dir)  # ai_document_assistant directory
sys.path.insert(0, project_root)

# Import the parent class with absolute import
from app.models.rag_model import RAGModel

# LangChain imports
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.memory import ConversationBufferMemory


class ChatRAGModel(RAGModel):
    def __init__(self):
        super().__init__()

        # Enhanced memory management
        self.conversation_memories: Dict[str, ConversationBufferMemory] = {}

        # Enhanced prompts for better conversation flow
        self.enhanced_contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", """Given a chat history and the latest user question which might reference 
                    context in the chat history, formulate a standalone question which can be understood 
                    without the chat history. 

                    Consider:
                    - Previous questions and answers in the conversation
                    - References to "it", "this", "that", "the document", etc.
                    - Follow-up questions that build on previous answers

                    Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])

        self.enhanced_qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant helping users understand documents. Use the following context 
                    to answer the question accurately and conversationally.

                    Guidelines:
                    1. Answer based ONLY on the provided context
                    2. If information is not in the context, say "I cannot find that information in the document"
                    3. Reference previous parts of the conversation when relevant
                    4. Be conversational but accurate
                    5. Quote specific parts of the document when helpful

                    Context: {context}"""),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])

    def get_conversation_memory(self, session_id: str) -> ConversationBufferMemory:
        """Get or create conversation buffer memory for a session."""
        if session_id not in self.conversation_memories:
            self.conversation_memories[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"  # Important for retrieval chains
            )
        return self.conversation_memories[session_id]

    def create_enhanced_chat_chain(self):
        """Create enhanced conversational retrieval chain with better memory."""
        if self.vector_store_manager.vector_store:
            # Create history-aware retriever with enhanced prompt
            self.history_aware_retriever = create_history_aware_retriever(
                self.llm,
                self.vector_store_manager.vector_store.as_retriever(
                    search_kwargs={"k": 6}  # Get more context for better answers
                ),
                self.enhanced_contextualize_prompt,
            )

            # Create document chain with enhanced prompt
            document_chain = create_stuff_documents_chain(
                self.llm,
                self.enhanced_qa_prompt
            )

            # Create the main retrieval chain
            self.retrieval_chain = create_retrieval_chain(
                self.history_aware_retriever,
                document_chain
            )

            return True
        return False

    def process_document(self, file_path: str) -> str:
        """Enhanced document processing with better chain creation."""
        try:
            # Process document using parent class
            chunks = self.document_processor.load_pdf(file_path)
            self.vector_store_manager.create_vector_store(chunks)

            # Create enhanced chat chain
            if self.create_enhanced_chat_chain():
                return f"Successfully processed document with {len(chunks)} chunks and enhanced chat capabilities."
            else:
                return f"Document processed ({len(chunks)} chunks) but chat chain creation failed."

        except Exception as e:
            return f"Error processing document: {str(e)}"

    def ask_question_with_memory(self, question: str, session_id: str = "default") -> Dict[str, Any]:
        """Enhanced question asking with conversation buffer memory."""
        if self.retrieval_chain is None:
            return {
                "answer": "Please upload and process a document first.",
                "sources": [],
                "session_id": session_id,
                "memory_used": False
            }

        try:
            # Get conversation memory for this session
            memory = self.get_conversation_memory(session_id)

            # Get chat history from both our custom storage and memory buffer
            custom_history = self.get_session_history(session_id).messages

            # Invoke the retrieval chain with chat history
            result = self.retrieval_chain.invoke({
                "input": question,
                "chat_history": custom_history
            })

            # Update both our custom history and the memory buffer
            history = self.get_session_history(session_id)
            history.add_user_message(question)
            history.add_ai_message(result["answer"])

            # Also update the conversation buffer memory
            memory.chat_memory.add_user_message(question)
            memory.chat_memory.add_ai_message(result["answer"])

            # Get sources from similarity search
            relevant_docs = self.vector_store_manager.similarity_search(question, k=4)

            return {
                "answer": result["answer"],
                "sources": [doc.page_content[:200] + "..." for doc in relevant_docs],
                "session_id": session_id,
                "memory_used": True,
                "conversation_length": len(custom_history) // 2
            }

        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "session_id": session_id,
                "memory_used": False
            }

    def get_memory_summary(self, session_id: str = "default") -> Dict[str, Any]:
        """Get a summary of the conversation memory."""
        if session_id in self.conversation_memories:
            memory = self.conversation_memories[session_id]
            messages = memory.chat_memory.messages

            return {
                "session_id": session_id,
                "message_count": len(messages),
                "has_memory": True,
                "last_interaction": messages[-1].content if messages else None
            }
        else:
            return {
                "session_id": session_id,
                "message_count": 0,
                "has_memory": False,
                "last_interaction": None
            }

    def clear_conversation_memory(self, session_id: str = "default") -> str:
        """Clear both custom history and conversation buffer memory."""
        # Clear custom history (parent class method)
        result = self.clear_chat_history(session_id)

        # Clear conversation buffer memory
        if session_id in self.conversation_memories:
            self.conversation_memories[session_id].clear()
            return f"{result} Conversation buffer memory also cleared."

        return result

    def export_conversation(self, session_id: str = "default") -> Dict[str, Any]:
        """Export the entire conversation with memory context."""
        custom_history = self.get_chat_history(session_id)
        memory_summary = self.get_memory_summary(session_id)

        return {
            "session_id": session_id,
            "conversation_history": custom_history,
            "memory_summary": memory_summary,
            "total_exchanges": len(custom_history) // 2 if custom_history else 0
        }
