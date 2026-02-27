from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
import torch
from torchvision import transforms, models
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import json
import uuid
import os
import shutil
import PIL.Image as Image
import io

print("Hello, FastAPI!")

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ========================= 모델 불러오기 =========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    global model
    model = models.resnet34(pretrained=True) # 모델 불러오기
    model.fc = nn.Linear(512, 4) # 모델의 마지막 레이어를 4개의 클래스로 변경
    print("Model loaded successfully!")
    print(f"Using device: {device}")

    # pth 파일이 딕셔너리 형태로 저장되어 있는 경우(체크포인트)와 모델의 state_dict(가중치)만 저장되어 있는 경우를 모두 처리
    checkpoint = torch.load("resnet34_transfer.pth", map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state dict loaded from checkpoint.")
    else:
        model.load_state_dict(checkpoint)
        print("Model state dict loaded directly from file.")
    model.to(device)
    model.eval()
    app.state.model = model
    app.state.device = device

    print("Model loaded successfully!")
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

# ========================= transforms 정의 =========================
transform_test = transforms.Compose([
    transforms.Resize((224, 224)), # 이미지 크기를 224x224로 조정
    transforms.ToTensor(), # 이미지 데이터를 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet 데이터셋의 평균과 표준편차로 정규화
])

@app.get("/")
def read_root():
    return {"Hello": "World!!"}

@app.post("/infer") # body : form-data, post가 아닌 get을 사용할 경우 client 측에서 파일을 보낼 수 없다. get은 url에 데이터를 담아서 보내는 방식이기 때문에 파일을 보낼 수 없다. post는 body에 데이터를 담아서 보내는 방식이기 때문에 파일을 보낼 수 있다.
async def infer(file: UploadFile = File(...)):
    model = app.state.model
    device = app.state.device
    allowed_ext = ['jpg', 'jpeg', 'png', 'webp']
    ext = file.filename.split(".")[-1].lower()

    if ext not in allowed_ext:
        return {"error" : "Invalid file type. Only .jpg, .jpeg, .png, and .webp are allowed."}
    
    newfile_name = f"{uuid.uuid4()}.{ext}"
    file_path = os.path.join("upload_img", newfile_name)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    #---------------------추론하기---------------------

    # img = await file.read() # 업로드된 파일의 내용을 읽음 f"upload_img/{file.filename}"
    img_data = Image.open(file_path).convert('RGB') # 업로드된 이미지를 열어서 RGB로 변환
    input_tensor = transform_test(img_data).unsqueeze(0).to(device) # 이미지에 대한 전처리를 수행하고 배치 차원을 추가한 후, 모델이 있는 디바이스로 이동

    with torch.no_grad(): # 추론 시에는 그래디언트 계산이 필요 없으므로 no_grad() 컨텍스트 매니저를 사용하여 메모리 사용을 최적화
        pred = model(input_tensor) # 모델을 사용하여 예측 수행 --> [0.6, 0.1, 0.2, 0.1]과 같은 형태의 출력이 나옴
        result = torch.argmax(pred, dim=1).item() # 예측 결과에서 가장 높은 확률을 가진 클래스의 인덱스를 추출
        # 클래스 인덱스를 클래스 이름으로 변환
        model_class = {
        "0" : "Amanda Seyfried",
        "1" : "Robert Downey Jr",
        "2" : "고윤정",
        "3" : "침착맨"
        }
        pred_class = model_class[str(result)]
        print(f'Predicted class: {pred_class} (index: {result})') # 예측된 클래스와 인덱스를 출력

# 사용자 데이터는 저장한다
# 향후 재사용 함을 공지한다
# 데이터 재활용 계획을 ppt로 만들어서 공지한다

    return {"result": pred_class, "index" : result, "filename": newfile_name, "model_class": model_class[str(result)]}

    # 데이터 수집 --> 전처리 --> 학습 --> 테스트 --> 테스트 --> 추론 --> 피드백(--> 데이터 수집)
