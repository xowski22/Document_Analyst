import streamlit as st
import requests
import os

MAX_FILE_SIZE = 1024 * 1024 * 10
SUPPORTED_FORMATS = ['.pdf', '.txt', '.docx']

def validate_file_frontend(uploaded_file):
    """Frontend validation of uploaded file"""
    if uploaded_file is None:
        return False, "No file uploaded"

    file_ext = os.path.splitext(uploaded_file.name)[1][1:].lower()

    if file_ext not in SUPPORTED_FORMATS:
        return False, f"Unsupported file format {file_ext}. Please choose from {SUPPORTED_FORMATS}"

    if uploaded_file.size > MAX_FILE_SIZE:
        return False, f"File too large. Maximum file size is {MAX_FILE_SIZE/1024/1024:.1f} MB"

    return True, ""

st.title("Document Analyzer")
uploaded_file = st.file_uploader("Choose a document...", type=SUPPORTED_FORMATS)

if uploaded_file is not None:

    is_valid, error_message = validate_file_frontend(uploaded_file)

    if not is_valid:
        st.error(error_message)
    else:
        with st.spinner('Analyzing document...'):
            try:
                files = {"file": uploaded_file}
                response = requests.post('http://localhost:8000/summarize', files=files)

                if response.status_code == 200:
                    st.subheader("Document Summary")
                    st.write(response.json()["summary"])
                else:
                    error_detail = response.json().get("detail", "Unknown error")
                    st.error(f"Error processing document: {error_detail}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to server. Please make sure the backend is running.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")


