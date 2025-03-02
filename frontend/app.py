import streamlit as st
import requests
import os
import hashlib
import json
from pathlib import Path
from src.utils.document_analyzer import document_analytics_tab

MAX_FILE_SIZE = 1024 * 1024 * 10
SUPPORTED_FORMATS = ['.pdf', '.txt', '.docx']
CACHE_DIR = Path("cache")

def setup_cache():
    """Setup cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_file_hash(file_bytes):
    """Generate hash for the file content"""
    return hashlib.md5(file_bytes).hexdigest()

def get_cached_summary(file_hash):
    """Try to get cached summary"""
    cache_file = CACHE_DIR / f"{file_hash}.json"
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)['summary']
    return None

def cache_summary(file_hash, summary):
    """Cache the summary"""
    cache_file = CACHE_DIR / f"{file_hash}.json"
    with open(cache_file, 'w') as f:
        json.dump({'summary': summary}, f)

def validate_file_frontend(uploaded_file):
    """Frontend validation of uploaded file"""
    if uploaded_file is None:
        return False, "No file uploaded"

    file_ext = os.path.splitext(uploaded_file.name)[1].lower()

    if file_ext not in SUPPORTED_FORMATS:
        return False, f"Unsupported file format {file_ext}. Please choose from {SUPPORTED_FORMATS}"

    if uploaded_file.size > MAX_FILE_SIZE:
        return False, f"File too large. Maximum file size is {MAX_FILE_SIZE/1024/1024:.1f} MB"

    return True, ""

def document_summary_section():
    uploaded_file = st.file_uploader("Choose a document...", type=["pdf", "txt", "docx"])

    if uploaded_file is not None:

        is_valid, error_message = validate_file_frontend(uploaded_file)

        if not is_valid:
            st.error(error_message)
        else:

            progress_bar = st.progress(0)
            status_text = st.empty()

            file_bytes = uploaded_file.getvalue()
            file_hash = get_file_hash(file_bytes)

            cached_summary = get_cached_summary(file_hash)

            if cached_summary:
                st.success("Retrieved from cache!")
                st.subheader("Document Summary:")
                st.write(cached_summary)
            else:
                try:

                    status_text.text("Starting document analysis...")
                    progress_bar.progress(10)

                    files = {"file": uploaded_file}

                    status_text.text("Sending document to server...")
                    progress_bar.progress(30)

                    response = requests.post('http://localhost:8000/summarize', files=files)

                    if response.status_code == 200:
                        progress_bar.progress(90)
                        status_text.text("Finalizing summary...")

                        summary = response.json()['summary']

                        cache_summary(file_hash, summary)

                        progress_bar.progress(100)
                        status_text.text("Analysis complete!")

                        st.subheader("Document Summary")
                        st.write(summary)
                    else:
                        error_detail = response.json().get("detail", "Unknown error")
                        st.error(f"Error processing document: {error_detail}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to server. Please make sure the backend is running.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
                finally:
                    status_text.empty()

def qa_section():
    """Question Answering section of Streamlit app"""
    st.subheader("Question Answering")

    input_method = st.radio(
        "Choose context input method:",
        ["Upload Document", "Enter Text"]
    )

    context_file = None
    context_text = None

    if input_method == "Upload Document":
        context_file = st.file_uploader(
            "Upload context document",
            type=["pdf", "txt", "docx"],
            key="qa_file_uploader"
        )

        if context_file:
            st.success(f"Uploaded: {context_file.name}")

    else:
        context_text = st.text_area(
            "Enter context text",
            height=200,
            key="qa_context_text",
            placeholder="Paste or type your text here"
        )

    question = st.text_input("Enter your question:", placeholder="Ask a question about the document...")

    if st.button("Get Answer"):
        if not question:
            st.error("Please enter a question.")
            return

        if not context_file and not context_text:
            st.error("Please provide context either through file upload or text input.")
            return

        try:
            with st.spinner("Finding answer..."):
                files = {}
                data = {
                    "question": question
                }

                if context_file:
                    files["context_file"] = context_file
                else:
                    data["context_text"] = context_text

                response = requests.post(
                    "http://localhost:8000/qa/ask",
                    data=data,
                    files=files
                )

                if response.status_code == 200:
                    result = response.json()

                    st.success("Answer found!")

                    st.markdown("### Answer")
                    st.markdown(f"**{result['answer']}**")

                    if "context_used" in result:
                        with st.expander("View source context"):
                            st.markdown("*Excerpt from document:*")
                            st.markdown(f"_{result["context_used"]}_")
                    else:
                        error_detail = response.json().get("detail", "Unknown error")
                        st.error(f"Error getting answer: {error_detail}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to server. Please make sure backend is running.")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")

def main():

    st.title("Document Analyzer")

    setup_cache()

    tab1, tab2, tab3 = st.tabs(["Document Summary", "Question Answering", "Document Analytics"])

    with tab1:
        document_summary_section()
    with tab2:
        qa_section()
    with tab3:
        document_analytics_tab()

if __name__ == "__main__":
    main()