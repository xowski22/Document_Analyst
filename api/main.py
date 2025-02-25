from fastapi import FastAPI, UploadFile, HTTPException, Form, File
from fastapi.responses import JSONResponse

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.summarizer import DocumentSummarizer
from src.utils.document_parser import DocumentParser
from src.models.qa_model import QuestionAnswerer

import tempfile
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor
import asyncio
from pydantic import BaseModel
from typing import Optional

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
qa_model = QuestionAnswerer()

MAX_FILE_SIZE = 1024 * 1024 * 10
SUPPORTED_FORMATS = ['.pdf', '.txt', '.docx']
MAX_WORKERS = 3

async def process_chunk(chunk: str) -> str:
    """Process a single chunk of text"""
    try:
        return summarizer.summarize(chunk)
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")
        return ""

async def process_chunks(chunks: List[str]) -> List[str]:
    """Process multiple chunks concurrently"""
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, summarizer.summarize, chunk)
            for chunk in chunks
        ]
        return await asyncio.gather(*tasks)

def validate_file(file: UploadFile) -> None:
    """Validate uploaded file format and size"""
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in SUPPORTED_FORMATS and file_ext.lstrip('.') not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats are: {', '.join([f.lstrip('.') for f in SUPPORTED_FORMATS])}"
        )

    try:
        content = file.file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size allowed is {MAX_FILE_SIZE/1024/1024:.1f}MB"
            )
        file.file.seek(0)

    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        raise HTTPException(status_code=400, detail="Error reading file")

@app.post("/qa/ask")
async def answer_question(
        question: str = Form(...),
        context_file: Optional[UploadFile] = File(None),
        context_text: Optional[str] = Form(None)
):
    """Answer a question based on either uploaded document or provided context."""

    try:
        if not context_file and not context_text:
            raise HTTPException(
                status_code=400,
                detail="Either context_file or context_text must be provided"
            )
        if context_file and context_text:
            raise HTTPException(
                status_code=400,
                detail="Please provide either context_file or context_text, not both"
            )

        if context_file:
            try:
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    content = await context_file.read()
                    temp_file.write(content)
                    temp_path = temp_file.name

                try:
                    context = parser.read_file(temp_path, original_filename=context_file.filename)
                    if not context or len(context.strip()) == 0:
                        raise ValueError("Empty document.")

                    context = parser.clean_text(context)

                finally:
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        logger.error(f"Error deleting temp file: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing file: {str(e)}")
                raise HTTPException(
                    status_code=400,
                    detail="Error processing uploaded file"
                )
        else:
            context = context_text

        try:
            answer = qa_model.answer_question(question, context)

            if not answer or answer == "Unable to find answer.":
                return JSONResponse(
                    status_code=200,
                    content={"answer": "Could not find answer in provided context.",
                             "confidence": 0.0
                             }
                )
            return {
                "answer": answer,
                "context_used": context[:200] + "..." if len(context) > 200 else context,
            }
        except Exception as e:
            logger.error(f"Error during question answering: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Error processing question"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred"
        )
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
            text = parser.read_file(temp_path, original_filename=file.filename)
            if not text or len(text.strip()) == 0:
                raise ValueError("Empty document")

            clean_text = parser.clean_text(text)
            chunks = summarizer.chunk_text(clean_text)

            if not chunks:
                raise ValueError("No content to summarize")


            # summaries: List[str] = []
            #
            # for i, chunk in enumerate(chunks):
            #     try:
            #         summary = summarizer.summarize(chunk)
            #         summaries.append(summary)
            #         logger.debug(f"Processed chunk {i+1}/{len(chunks)}")
            #     except Exception as e:
            #         logger.error(f"Error summarizing chunk {i+1}/{str(e)}")
            #         continue
            #
            # if not summaries:
            #     raise ValueError("Failed to generate summary")

            chunk_summaries = await process_chunks(chunks)

            valid_summaries = [s for s in chunk_summaries if s]
            if not valid_summaries:
                raise ValueError("Failed to generate summary")

            final_summary = " ".join(valid_summaries)
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

