import streamlit as st
import requests

st.title("Document Analyzer")
uploaded_file = st.file_uploader("Choose a document...", type=['pdf', 'docx', 'txt'])

if uploaded_file is not None:
    with st.spinner('Analyzing document...'):
        files = {"file": uploaded_file}
        response = requests.post('http://localhost:8000/summarize', files=files)

        if response.status_code == 200:
            st.subheader("Document Summary")
            st.write(response.json()["summary"])
        else:
            st.error("Error processing document")