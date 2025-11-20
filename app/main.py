import warnings
import os
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('torch').setLevel(logging.ERROR)

import streamlit as st
import tempfile
import uuid
import sys
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF

# Add project root to sys.path
current_file = os.path.abspath(__file__)
app_dir = os.path.dirname(current_file)
project_root = os.path.dirname(app_dir)
sys.path.insert(0, project_root)

from app.models.rag_model import RAGModel

# Page configuration
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'rag_model' not in st.session_state:
    st.session_state.rag_model = RAGModel()
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if 'current_answer' not in st.session_state:
    st.session_state.current_answer = None
if 'chat_history_display' not in st.session_state:
    st.session_state.chat_history_display = None
if 'comparison_mode' not in st.session_state:
    st.session_state.comparison_mode = False
if 'comparison_files' not in st.session_state:
    st.session_state.comparison_files = []
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None
if 'structured_data' not in st.session_state:
    st.session_state.structured_data = None
if 'recommendation' not in st.session_state:
    st.session_state.recommendation = None
if 'last_uploaded_files' not in st.session_state:
    st.session_state.last_uploaded_files = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# Custom CSS
st.markdown("""
<style>
.question-display, .question-display h4, .question-display p {
    color: #0f172a !important;
}

.main-answer, .main-answer h4, .main-answer p {
    color: #0f172a !important;
}

.question-display {
    background-color: #e6f0f7 !important;
    border: 2px solid #1e88e5;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
}

.main-answer {
    background-color: #f7fff8 !important;
    border: 2px solid #2e7d32;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
}

.question-display div, .question-display span, .main-answer div, .main-answer span {
    color: #0f172a !important;
}

.comparison-card {
    background-color: #fef9e7;
    border: 2px solid #f39c12;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
}

.comparison-card h4, .comparison-card p {
    color: #0f172a !important;
}

div[data-testid="stExpander"] * {
    color: #e5e7eb !important;
}
</style>
""", unsafe_allow_html=True)


# ==================== VISUALIZATION HELPERS ====================

def create_skills_comparison_chart(structured_data, doc_ids):
    """Create bar chart comparing skills count"""
    skills_count = {}

    for doc_id in doc_ids:
        if doc_id in structured_data:
            data = structured_data[doc_id]
            doc_name = st.session_state.rag_model.documents[doc_id]['filename']
            skills = data.get('skills', [])
            if isinstance(skills, list):
                skills_count[doc_name] = len(skills)
            else:
                skills_count[doc_name] = 0

    if not skills_count:
        return None

    fig = go.Figure(data=[
        go.Bar(
            x=list(skills_count.keys()),
            y=list(skills_count.values()),
            marker_color=['#1e88e5', '#43a047', '#fb8c00'][:len(skills_count)],
            text=list(skills_count.values()),
            textposition='auto',
        )
    ])

    fig.update_layout(
        title="Skills Count Comparison",
        xaxis_title="Document",
        yaxis_title="Number of Skills",
        height=400,
        showlegend=False,
        hovermode='x'
    )

    return fig


def create_experience_comparison(structured_data, doc_ids):
    """Create bar chart comparing years of experience"""
    experience_data = {}

    for doc_id in doc_ids:
        if doc_id in structured_data:
            data = structured_data[doc_id]
            doc_name = st.session_state.rag_model.documents[doc_id]['filename']
            exp_years = data.get('experience_years', 0)
            try:
                experience_data[doc_name] = int(exp_years) if exp_years else 0
            except:
                experience_data[doc_name] = 0

    if not experience_data:
        return None

    fig = go.Figure(data=[
        go.Bar(
            x=list(experience_data.keys()),
            y=list(experience_data.values()),
            marker_color=['#26a69a', '#5c6bc0', '#ef5350'][:len(experience_data)],
            text=[f"{y} years" for y in experience_data.values()],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title="Experience Comparison (Years)",
        xaxis_title="Candidate",
        yaxis_title="Years of Experience",
        height=400,
        showlegend=False
    )

    return fig


def create_document_size_comparison(doc_ids):
    """Create comparison of document sizes"""
    size_data = {
        'Document': [],
        'Words': [],
        'Characters': [],
        'Sections': []
    }

    for doc_id in doc_ids:
        doc_info = st.session_state.rag_model.documents[doc_id]
        size_data['Document'].append(doc_info['filename'])
        size_data['Words'].append(doc_info['stats']['total_words'])
        size_data['Characters'].append(doc_info['stats']['total_characters'])
        size_data['Sections'].append(doc_info['stats']['total_chunks'])

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Words',
        x=size_data['Document'],
        y=size_data['Words'],
        marker_color='#42a5f5'
    ))

    fig.add_trace(go.Bar(
        name='Sections',
        x=size_data['Document'],
        y=size_data['Sections'],
        marker_color='#66bb6a',
        yaxis='y2'
    ))

    fig.update_layout(
        title="Document Size Comparison",
        xaxis_title="Document",
        yaxis_title="Words",
        yaxis2=dict(
            title="Sections",
            overlaying='y',
            side='right'
        ),
        barmode='group',
        height=400,
        hovermode='x unified'
    )

    return fig


# ==================== PDF EXPORT HELPER ====================

def generate_comparison_pdf(doc_ids, comparison_results, recommendation=None):
    """Generate PDF report of comparison"""

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Document Comparison Report', ln=True, align='C')

    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', ln=True, align='C')
    pdf.ln(10)

    # Documents compared
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Documents Compared:', ln=True)
    pdf.set_font('Arial', '', 10)

    for doc_id in doc_ids:
        doc_name = st.session_state.rag_model.documents[doc_id]['filename']
        # Truncate long filenames
        if len(doc_name) > 50:
            doc_name = doc_name[:47] + '...'
        pdf.cell(0, 8, f'  - {doc_name}', ln=True)

    pdf.ln(5)

    # Statistics
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Document Statistics:', ln=True)
    pdf.set_font('Arial', '', 9)

    for doc_id in doc_ids:
        doc_info = st.session_state.rag_model.documents[doc_id]
        doc_name = doc_info['filename']
        if len(doc_name) > 40:
            doc_name = doc_name[:37] + '...'

        pdf.cell(0, 6, f"{doc_name}:", ln=True)
        pdf.cell(0, 5, f"  Words: {doc_info['stats']['total_words']:,}", ln=True)
        pdf.cell(0, 5, f"  Sections: {doc_info['stats']['total_chunks']}", ln=True)
        pdf.ln(2)

    pdf.ln(5)

    # Comparison results
    if comparison_results:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Detailed Comparison:', ln=True)

        for aspect, candidates in comparison_results.items():
            # Check if we need a new page
            if pdf.get_y() > 250:
                pdf.add_page()

            pdf.set_font('Arial', 'B', 11)
            aspect_text = aspect.title()
            if len(aspect_text) > 60:
                aspect_text = aspect_text[:57] + '...'
            pdf.multi_cell(0, 8, f'\n{aspect_text}:')

            pdf.set_font('Arial', '', 8)  # Smaller font for content

            for doc_id in doc_ids:
                doc_name = st.session_state.rag_model.documents[doc_id]['filename']
                if len(doc_name) > 30:
                    doc_name = doc_name[:27] + '...'

                content = candidates.get(doc_id, 'N/A')

                # Clean and truncate content
                content = content.replace('\n', ' ').replace('\r', ' ')
                content = ' '.join(content.split())  # Remove extra spaces

                # Limit content length
                if len(content) > 250:
                    content = content[:247] + '...'

                # Safe multi-cell with error handling
                try:
                    pdf.set_left_margin(15)
                    pdf.multi_cell(0, 5, f'{doc_name}: {content}', align='L')
                    pdf.set_left_margin(10)
                except Exception as e:
                    # Fallback: use cell instead
                    pdf.cell(0, 5, f'{doc_name}: [Content too long]', ln=True)

                pdf.ln(2)

    # Recommendation
    if recommendation:
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'AI Recommendation:', ln=True)
        pdf.set_font('Arial', '', 9)

        # Clean recommendation text
        rec_text = recommendation.replace('\n\n', '\n').replace('\r', '')
        rec_text = rec_text[:1500]  # Limit length

        try:
            pdf.multi_cell(0, 5, rec_text)
        except Exception as e:
            # Fallback for very long text
            pdf.cell(0, 5, 'Recommendation available in the app', ln=True)

    try:
        return pdf.output(dest='S').encode('latin1')
    except:
        # If encoding fails, try UTF-8
        return pdf.output(dest='S').encode('utf-8', errors='ignore')


# ==================== NAVIGATION ====================
st.sidebar.title("🤖 AI Document Assistant")
mode = st.sidebar.radio(
    "Select Mode:",
    ["📄 Single Document Q&A", "⚖️ Document Comparison"],
    key="mode_selector"
)

# ==================== SINGLE DOCUMENT MODE ====================
if mode == "📄 Single Document Q&A":
    st.session_state.comparison_mode = False

    st.title("🤖 AI Document Assistant")
    st.markdown("Upload a PDF document and ask questions about its content!")

    with st.sidebar:
        st.header("📁 Upload Document")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document to analyze"
        )

        if uploaded_file is not None:
            if st.button("Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    result = st.session_state.rag_model.process_document(tmp_file_path)
                    os.unlink(tmp_file_path)
                    st.success(result)
                    st.session_state.document_processed = True

        if st.session_state.document_processed:
            st.header("💬 Chat Management")
            st.write(f"**Current Session:** {st.session_state.session_id}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🆕 New Session"):
                    st.session_state.session_id = str(uuid.uuid4())[:8]
                    st.session_state.current_answer = None
                    st.success(f"New session: {st.session_state.session_id}")
                    st.rerun()

            with col2:
                if st.button("🗑️ Clear Current"):
                    st.session_state.rag_model.clear_chat_history(st.session_state.session_id)
                    st.session_state.current_answer = None
                    st.success("Chat cleared!")
                    st.rerun()

            st.subheader("📚 Chat Sessions")
            active_sessions = st.session_state.rag_model.list_active_sessions()

            if active_sessions:
                session_options = ["Select a session..."]
                session_mapping = {}

                for session in active_sessions:
                    session_history = st.session_state.rag_model.get_chat_history(session)
                    chat_count = len(session_history) // 2 if session_history else 0

                    if session == st.session_state.session_id:
                        display_name = f"🟢 Session {session} ({chat_count} Q&As) - Current"
                    else:
                        display_name = f"⚪ Session {session} ({chat_count} Q&As)"

                    session_options.append(display_name)
                    session_mapping[display_name] = session

                selected_display = st.selectbox(
                    "Choose Chat History:",
                    options=session_options,
                    index=0
                )

                if selected_display != "Select a session..." and selected_display in session_mapping:
                    selected_session_id = session_mapping[selected_display]

                    col_action1, col_action2 = st.columns(2)
                    with col_action1:
                        if st.button("🔄 Follow Up", key=f"followup_{selected_session_id}"):
                            st.session_state.session_id = selected_session_id
                            session_history = st.session_state.rag_model.get_chat_history(selected_session_id)
                            if session_history:
                                st.session_state.chat_history_display = session_history
                                st.success(f"✅ Loaded Session: {selected_session_id}")
                            st.rerun()

                    with col_action2:
                        if st.button("🗑️ Clear", key=f"clear_{selected_session_id}"):
                            st.session_state.rag_model.clear_chat_history(selected_session_id)
                            if selected_session_id == st.session_state.session_id:
                                st.session_state.current_answer = None
                            st.success(f"🗑️ Cleared: {selected_session_id}")
                            st.rerun()

            st.subheader("📊 Statistics")
            total_sessions = len(active_sessions) if active_sessions else 0
            current_session_history = st.session_state.rag_model.get_chat_history(st.session_state.session_id)
            current_chat_count = len(current_session_history) // 2 if current_session_history else 0
            st.metric("Total Sessions", total_sessions)
            st.metric("Current Q&As", current_chat_count)

    if st.session_state.document_processed:
        st.header("💬 Ask Questions")

        current_session_history = st.session_state.rag_model.get_chat_history(st.session_state.session_id)
        current_chat_count = len(current_session_history) // 2 if current_session_history else 0
        st.info(f"**Session:** {st.session_state.session_id} | **Questions:** {current_chat_count}")

        with st.form(key="question_form", clear_on_submit=True):
            question = st.text_area(
                "What would you like to know?",
                placeholder="e.g., What is the main topic?",
                height=120
            )
            ask_button = st.form_submit_button("Ask Question", type="primary")

        if ask_button and question.strip():
            st.session_state.current_answer = None
            with st.spinner("Generating answer..."):
                response = st.session_state.rag_model.ask_question(question, st.session_state.session_id)


                def clean_ai_response(text):
                    text = text.replace("System:", "").replace("Human:", "").replace("Assistant:", "")
                    text = text.replace("Answer:", "").replace("Response:", "")
                    lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 5]
                    return ' '.join(lines).strip()


                clean_answer = clean_ai_response(response["answer"])
                if not clean_answer or len(clean_answer) < 5:
                    clean_answer = "I cannot provide a clear answer."

                st.session_state.current_answer = {
                    "question": question,
                    "answer": clean_answer,
                    "sources": response.get("sources", [])
                }

        if st.session_state.current_answer:
            st.markdown("---")
            st.markdown(f"""
            <div class="question-display">
                <h4>❓ Your Question:</h4>
                <p>{st.session_state.current_answer['question']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="main-answer">
                <h4>🤖 Answer:</h4>
                <p>{st.session_state.current_answer['answer']}</p>
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.current_answer['sources']:
                with st.expander("📚 View Sources", expanded=False):
                    for i, source in enumerate(st.session_state.current_answer['sources'], 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(source)

        if st.session_state.get('chat_history_display'):
            st.markdown("---")
            st.subheader("📜 Complete Chat History")
            for i, message in enumerate(st.session_state.chat_history_display):
                if message["role"] == "user":
                    q_num = (i // 2) + 1
                    st.markdown(f"""
                    <div class="question-display">
                        <h4>❓ Question {q_num}:</h4>
                        <p>{message["content"]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif message["role"] == "assistant":
                    a_num = (i // 2) + 1
                    st.markdown(f"""
                    <div class="main-answer">
                        <h4>🤖 Answer {a_num}:</h4>
                        <p>{message["content"]}</p>
                    </div>
                    """, unsafe_allow_html=True)

            if st.button("🔄 Hide History", type="secondary"):
                del st.session_state.chat_history_display
                st.rerun()

    else:
        st.header("💬 Get Started")
        st.info("👆 Upload a PDF in the sidebar to begin!")
        st.markdown("""
        ### Welcome!
        - 📄 Upload PDFs
        - 🤖 Ask questions
        - 💬 Multi-session chats
        - ⚖️ Compare documents
        """)

# ==================== COMPARISON MODE ====================
elif mode == "⚖️ Document Comparison":
    st.session_state.comparison_mode = True

    # Add reset button
    if st.session_state.comparison_files:
        col_title, col_reset = st.columns([5, 1])
        with col_title:
            st.title("⚖️ Smart Document Comparison")
        with col_reset:
            if st.button("🔄 Reset", help="Clear all and start over"):
                st.session_state.comparison_files = []
                st.session_state.last_uploaded_files = []
                st.session_state.comparison_results = None
                st.session_state.structured_data = None
                st.session_state.recommendation = None
                st.session_state.processing_complete = False
                st.rerun()
    else:
        st.title("⚖️ Smart Document Comparison")

    st.markdown("Upload 2-3 documents (resumes, reports, etc.) to compare side-by-side")

    with st.sidebar:
        st.header("📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose 2-3 PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload 2-3 documents to compare",
            key="comparison_uploader"
        )

        if uploaded_files and len(uploaded_files) >= 2:
            # Get current file names
            current_files = [f.name for f in uploaded_files]

            # Check if files changed
            files_changed = (current_files != st.session_state.last_uploaded_files)
            already_processed = st.session_state.processing_complete

            # Show process button only if not yet processed or files changed
            if not already_processed or files_changed:
                if st.button("🔄 Process & Compare", type="primary", key="process_btn"):
                    if len(uploaded_files) > 3:
                        st.error("Maximum 3 documents allowed")
                    else:
                        # Store current file names
                        st.session_state.last_uploaded_files = current_files

                        status_text = st.empty()

                        try:
                            status_text.info("📄 Extracting text from PDFs...")

                            temp_paths = []
                            filename_mapping = {}  # Map temp path to original filename

                            for uploaded_file in uploaded_files:
                                # Create temp file with original filename preserved
                                temp_dir = tempfile.gettempdir()
                                # Use original filename in temp directory
                                temp_path = os.path.join(temp_dir, uploaded_file.name)

                                # Write file
                                with open(temp_path, 'wb') as f:
                                    f.write(uploaded_file.getvalue())

                                temp_paths.append(temp_path)
                                filename_mapping[temp_path] = uploaded_file.name

                            status_text.info("🔄 Creating embeddings (30-60 seconds)...")
                            results = st.session_state.rag_model.process_multiple_documents(temp_paths)

                            # Clean up temp files
                            for path in temp_paths:
                                try:
                                    os.unlink(path)
                                except:
                                    pass

                            # Store results
                            st.session_state.comparison_files = list(results.keys())
                            st.session_state.processing_complete = True

                            # Clear status and show success
                            status_text.empty()
                            st.success(f"✅ Processed {len(results)} documents successfully!")

                        except Exception as e:
                            status_text.error(f"❌ Error: {str(e)}")
                            for path in temp_paths:
                                try:
                                    os.unlink(path)
                                except:
                                    pass

            # Show status if already processed
            elif already_processed and not files_changed:
                st.info(f"✅ {len(st.session_state.comparison_files)} documents ready")
                if st.button("🗑️ Clear & Upload New", key="clear_btn"):
                    st.session_state.comparison_files = []
                    st.session_state.last_uploaded_files = []
                    st.session_state.comparison_results = None
                    st.session_state.structured_data = None
                    st.session_state.recommendation = None
                    st.session_state.processing_complete = False
                    st.rerun()

    if st.session_state.comparison_files:
        doc_ids = st.session_state.comparison_files

        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Quick Stats",
            "🔍 Detailed Comparison",
            "💡 Recommendation",
            "📈 Visual Analytics"
        ])

        with tab1:
            st.subheader("📊 Document Statistics")

            stats_data = []
            for doc_id in doc_ids:
                doc_info = st.session_state.rag_model.documents[doc_id]
                stats_data.append({
                    'Document': doc_info['filename'],
                    'Words': f"{doc_info['stats']['total_words']:,}",
                    'Characters': f"{doc_info['stats']['total_characters']:,}",
                    'Sections': doc_info['stats']['total_chunks']
                })

            df_stats = pd.DataFrame(stats_data)
            st.dataframe(df_stats, use_container_width=True)

            st.markdown("### 📈 Visual Comparison")
            fig_size = create_document_size_comparison(doc_ids)
            st.plotly_chart(fig_size, use_container_width=True)

        with tab2:
            st.subheader("🔍 Detailed Side-by-Side Comparison")

            if st.button("🔄 Run Comparison Analysis", type="primary"):
                with st.spinner("Analyzing documents..."):
                    comparison_results = st.session_state.rag_model.compare_documents(doc_ids)
                    st.session_state.comparison_results = comparison_results

            if st.session_state.comparison_results:
                results = st.session_state.comparison_results

                for aspect, candidates in results.items():
                    st.markdown(f"### {aspect.title()}")

                    cols = st.columns(len(doc_ids))
                    for idx, doc_id in enumerate(doc_ids):
                        with cols[idx]:
                            doc_name = st.session_state.rag_model.documents[doc_id]['filename']
                            st.markdown(f"**{doc_name}**")
                            st.info(candidates.get(doc_id, "N/A"))

                    st.markdown("---")

        with tab3:
            st.subheader("💡 AI Recommendation")

            job_role = st.text_input(
                "Job Role (optional)",
                placeholder="e.g., Senior Data Analyst"
            )

            recommendation_text = None

            if st.button("Get Recommendation", type="primary"):
                with st.spinner("Generating recommendation..."):
                    recommendation_text = st.session_state.rag_model.get_recommendation(doc_ids, job_role)
                    st.session_state.recommendation = recommendation_text
                    st.markdown("### 🎯 Recommendation")
                    st.success(recommendation_text)

            # Display previously generated recommendation
            if 'recommendation' in st.session_state and st.session_state.recommendation:
                if not recommendation_text:
                    st.markdown("### 🎯 Previous Recommendation")
                    st.success(st.session_state.recommendation)

                # PDF Export Section
                st.markdown("---")
                st.markdown("### 📥 Export Report")

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.info("💡 Download a comprehensive PDF report with comparison analysis and recommendations")

                with col2:
                    if st.button("📄 Generate PDF", type="secondary"):
                        with st.spinner("Generating PDF..."):
                            pdf_data = generate_comparison_pdf(
                                doc_ids,
                                st.session_state.comparison_results if st.session_state.comparison_results else {},
                                st.session_state.recommendation
                            )

                            st.download_button(
                                label="📥 Download Report",
                                data=pdf_data,
                                file_name=f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                type="primary"
                            )
                            st.success("✅ PDF ready for download!")

        with tab4:
            st.subheader("📈 Visual Analytics")

            if st.button("Generate Visual Analysis", type="primary", key="generate_visuals"):
                with st.spinner("Extracting and visualizing data..."):
                    structured_data = {}
                    for doc_id in doc_ids:
                        structured_data[doc_id] = st.session_state.rag_model.extract_structured_data(doc_id)

                    st.session_state.structured_data = structured_data

            if st.session_state.structured_data:
                structured_data = st.session_state.structured_data

                # Skills comparison chart
                st.markdown("### 📊 Skills Count Comparison")
                fig_skills = create_skills_comparison_chart(structured_data, doc_ids)
                if fig_skills:
                    st.plotly_chart(fig_skills, use_container_width=True)
                else:
                    st.info("No skills data available for visualization")

                # Experience comparison
                st.markdown("### 💼 Experience Comparison")
                fig_exp = create_experience_comparison(structured_data, doc_ids)
                if fig_exp:
                    st.plotly_chart(fig_exp, use_container_width=True)
                else:
                    st.info("No experience data available for visualization")

                # Structured data display
                st.markdown("### 📄 Detailed Data")
                for doc_id in doc_ids:
                    data = structured_data[doc_id]
                    doc_name = st.session_state.rag_model.documents[doc_id]['filename']
                    with st.expander(f"📄 {doc_name}", expanded=False):
                        st.json(data)

    else:
        st.info("👆 Upload 2-3 documents in the sidebar to start comparing")

# Footer
st.markdown("---")
st.caption("🤖 AI Document Assistant")
