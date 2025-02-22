from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from src.models.summarizer import DocumentSummarizer
from src.utils.document_parser import DocumentParser
import tempfile
import os
import logging
from typing import List

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger(__name__)

app = FastAPI()
summarizer = DocumentSummarizer()
parser = DocumentParser()

MAX_FILE_SIZE = 1024 * 1024 * 10
SUPPORTED_FORMATS = ['.pdf', '.txt', '.docx']

def validate_file(file: UploadFile) -> None:
    """Validate uploaded file format and size"""
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats are: {', '.join(SUPPORTED_FORMATS)}"
        )

    try:
        content = file.file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size allowed is {MAX_FILE_SIZE/1024/1024:.1f}MB"
            )
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        raise HTTPException(status_code=400, detail="Error reading file")

@app.post("/summarize")
async def summarize_document(file: UploadFile):
    try:

        validate_file(file)
        logger.info(f"Processing file: {file.filename}")

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        try:
            text = parser.read_text(temp_path)
            if not text or len(text.strip()) == 0:
                raise ValueError("Empty document")

            clean_text = parser.clean_text(text)
            chunks = summarizer.summarize(clean_text)

            if not chunks:
                raise ValueError("No content to summarize")


            summaries: List[str] = []

            for i, chunk in enumerate(chunks):
                try:
                    summary = summarizer.summarize(chunk)
                    summaries.append(summary)
                    logger.debug(f"Processed chunk {i+1}/{len(chunks)}")
                except Exception as e:
                    logger.error(f"Error summarizing chunk {i+1}/{str(e)}")
                    continue

            if not summaries:
                raise ValueError("Failed to generate summary")

            final_summary = " ".join(summaries)
            logger.info(f"Successfully summarized document:  {file.filename}")

            return {"summary": final_summary}

        except ValueError as ve:
            logger.error(f"Validation error: {str(ve)}")
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            raise HTTPException(status_code=500, detail="Error processing document")
        finally:
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.error(f"Error error removing temporary file: {str(e)}")
    except Exception as e:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content="An unexpected error occurred")

