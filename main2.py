from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import json
import uuid
import os
import shutil

app = FastAPI()
model = None

@app.on_event('startup')
def load_model():
    print('========== 모델 불러오기 시작 ==============')
    global model
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(512,3)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load('best_model.pth',map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else :
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
print('========== 모델 불러오기 끝!!! ==============')

transform_test = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]
)



@app.get('/')
def root():
    return {"result" : "Hi!!!"}



@app.post('/infer')
def infer(file:UploadFile = File(...)):  #body : form-data

    allowed_ext = ["jpg","jpeg","png","webp"]
    ext = file.filename.split(".")[-1].lower()  #"a.png" -> split    ['a','png']

    if ext not in allowed_ext:
        return {'error':'이미지파일만 업로드하세요!!!!!'}

    newfile_name = f'{uuid.uuid4()}.{ext}'
    file_path = os.path.join('upload_img',newfile_name)

    with open(file_path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    #-----------------추론코드--------------------------

    import PIL.Image as Image
    img = Image.open(file_path)
    img_t = transform_test(img)
    batch_t = torch.unsqueeze(img_t,0)

    with torch.no_grad():
        outputs = model(batch_t)
        _, preds = torch.max(outputs,1)

    class_names = ['bts','jungkook','suga']
    result = class_names[preds[0]]
    return {"result" : result , "index" : str(preds[0].item()), "filename":newfile_name }