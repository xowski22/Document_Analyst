from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from src.models.summarizer import DocumentSummarizer
from src.utils.document_parser import DocumentParser
import tempfile
import os

app = FastAPI()
summarizer = DocumentSummarizer()
parser = DocumentParser()

@app.post("/summarize")
async def summarize_document(file: UploadFile):
    try:

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        text = parser.read_text(temp_path)
        clean_text = parser.clean_text(text)
        chunks = summarizer.summarize(clean_text)

        summaries = []

        for chunk in chunks:
            summary = summarizer.summarize(chunk)
            summaries.append(summary)

        final_summary = " ".join(summaries)

        os.unlink(temp_path)

        return {"summary": final_summary}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)})

