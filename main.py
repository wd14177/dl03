from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
from torchvision import transforms, models
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import json
import uuid
import os
import shutil

print("Hello, FastAPI!")

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World!!"}

@app.post("/infer") # body : form-data
def infer(file: UploadFile = File(...)):
    allowed_ext = ['jpg', 'jpeg', 'png', 'webp']
    ext = file.filename.split(".")[-1].lower()

    if ext not in allowed_ext:
        return {"error" : "Invalid file type. Only .jpg, .jpeg, .png, and .webp are allowed."}
    
    newfile_name = f"{uuid.uuid4()}.{ext}"
    file_path = os.path.join("upload_img", newfile_name)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

#---------------------추론하기---------------------
# 사용자 데이터는 저장한다
# 향후 재사용 함을 공지한다
    
    return {"result": "class1", "index" : "2", "filename": newfile_name}

    # 데이터 수집 --> 전처리 --> 학습 --> 테스트 --> 테스트 --> 추론 --> 피드백(--> 데이터 수집)
