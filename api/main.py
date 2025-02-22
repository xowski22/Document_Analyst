from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/summarize")
async def summarze_document(file: UploadFile):
    pass