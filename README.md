# 🤖 AI Document Assistant with RAG

An intelligent document analysis and comparison system powered by Retrieval Augmented Generation (RAG) architecture. Process single documents with natural language Q&A or compare multiple documents side-by-side with AI-generated insights.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

### 📄 Single Document Q&A Mode
- Upload any PDF document (resume, report, contract, research paper)
- Ask questions in natural language
- Get instant, context-aware answers using RAG
- Multi-session chat with conversation history
- Source citations for transparency

### ⚖️ Multi-Document Comparison Mode
- Upload 2-3 documents simultaneously
- AI-powered side-by-side comparison across:
  - Skills and technologies
  - Work experience and roles
  - Education and certifications
  - Key achievements
  - Strengths and areas for improvement
- Interactive data visualizations (Plotly charts)
- AI-generated recommendations
- Export comprehensive PDF reports

## 🎯 Use Cases

| Industry | Application |
|----------|-------------|
| **HR & Recruitment** | Screen and compare candidate resumes instantly |
| **Business Analysis** | Compare quarterly reports, proposals, contracts |
| **Research** | Analyze and compare academic papers |
| **Legal** | Review and compare contract versions |
| **Consulting** | Compare vendor proposals and RFPs |

## 🛠️ Tech Stack

- **LLM:** Groq (Llama 3.1 8B) - 10x faster than GPT-4 for RAG tasks
- **Framework:** LangChain - RAG pipeline orchestration
- **Vector Store:** FAISS - Fast similarity search (CPU-friendly)
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **UI:** Streamlit - Interactive dual-mode interface
- **Visualizations:** Plotly - Interactive charts
- **PDF Processing:** PyPDF2 - Document parsing
- **Export:** FPDF2 - PDF report generation

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- Groq API key (free tier available at https://console.groq.com)

### Installation

1. **Clone the repository**
