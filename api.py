from fastapi import FastAPI, UploadFile, File
from typing import List
import os

app = FastAPI()

UPLOAD_DIR = "to_be_processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)


#Allows the user to upload multiple pictures via API
#stores the pictures into the "to_be_processed" folder

@app.post("/upload/")
async def upload(file: UploadFile):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    content = await file.read()

    with open(file_path, "wb") as f:
        f.write(content)
    return {"uploaded_file": file.filename}