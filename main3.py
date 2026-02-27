from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from contextlib import asynccontextmanager
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision import transforms
from PIL import Image
import uuid
import os
import shutil
import numpy as np

# ----------------------------
# Global
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UPLOAD_DIR = "upload_img"
CAM_DIR = "gradcam_img"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CAM_DIR, exist_ok=True)

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 예시 라벨
model_class = {
    "0": "Amanda Seyfried",
    "1": "Robert Downey Jr",
    "2": "고윤정",
    "3": "침착맨",
}

# ----------------------------
# Grad-CAM (hook 기반)
# ----------------------------
class GradCAM:
    """
    ResNet 계열을 기준으로 마지막 conv feature map을 잡아서 Grad-CAM 생성
    target_layer: 보통 model.layer4[-1].conv2 (resnet34)
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out  # [B, C, H, W]

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]  # [B, C, H, W]

    def close(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    @torch.enable_grad()
    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None):
        """
        returns: cam (H, W) in [0,1]
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)  # [1, num_classes]

        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        # activations, gradients shape: [1, C, H, W]
        acts = self.activations.detach()
        grads = self.gradients.detach()

        # weights: GAP over spatial dims => [1, C, 1, 1]
        weights = grads.mean(dim=(2, 3), keepdim=True)

        # weighted sum: [1, 1, H, W]
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        # normalize to [0,1]
        cam = cam[0, 0]
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam.cpu().numpy(), class_idx


def overlay_cam_on_image(orig_pil: Image.Image, cam_01: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    orig_pil: 원본 (W,H)
    cam_01: [H',W'] 0~1
    """
    # cam을 원본 크기로 리사이즈
    cam_img = Image.fromarray(np.uint8(cam_01 * 255)).resize(orig_pil.size, resample=Image.BILINEAR)

    # 간단한 colormap (jet 비슷하게) 직접 생성
    cam_arr = np.array(cam_img).astype(np.float32) / 255.0  # [H,W]
    r = np.clip(1.5 * cam_arr - 0.5, 0, 1)
    g = np.clip(1.5 - 1.5 * np.abs(cam_arr - 0.5), 0, 1)
    b = np.clip(0.5 - 1.5 * cam_arr, 0, 1)
    heat = np.stack([r, g, b], axis=-1)  # [H,W,3]
    heat_pil = Image.fromarray(np.uint8(heat * 255)).convert("RGB")

    orig_rgb = orig_pil.convert("RGB")
    blended = Image.blend(orig_rgb, heat_pil, alpha=alpha)
    return blended


# ----------------------------
# FastAPI lifespan: 모델 로드
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    model = tv_models.resnet34(weights=tv_models.ResNet34_Weights.DEFAULT)
    model.fc = nn.Linear(512, 4)

    # checkpoint 로드 (있을 때만)
    ckpt_path = "resnet34_transfer.pth"
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Grad-CAM 대상 레이어 선택 (ResNet34 기준)
    target_layer = model.layer4[-1].conv2

    app.state.model = model
    app.state.target_layer = target_layer

    print("Model loaded.")
    yield
    print("Shutdown.")

app = FastAPI(lifespan=lifespan)


# ----------------------------
# 원본/캠 이미지 서빙
# ----------------------------
@app.get("/uploads/{filename}")
def get_uploaded(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    return FileResponse(path)

@app.get("/gradcam/{camname}")
def get_cam(camname: str):
    path = os.path.join(CAM_DIR, camname)
    if not os.path.exists(path):
        raise HTTPException(404, "CAM file not found")
    return FileResponse(path)


# ----------------------------
# 업로드 -> 원본 보여주기 + Grad-CAM 생성/표시
# ----------------------------
@app.post("/infer", response_class=HTMLResponse)
async def infer(file: UploadFile = File(...)):
    allowed_ext = {"jpg", "jpeg", "png", "webp"}
    ext = file.filename.split(".")[-1].lower()
    if ext not in allowed_ext:
        raise HTTPException(400, "Invalid file type")

    # 저장
    uid = str(uuid.uuid4())
    img_name = f"{uid}.{ext}"
    img_path = os.path.join(UPLOAD_DIR, img_name)
    with open(img_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 원본 로드
    orig = Image.open(img_path).convert("RGB")

    # 모델 입력 텐서
    input_tensor = transform_test(orig).unsqueeze(0).to(device)

    model: nn.Module = app.state.model
    target_layer: nn.Module = app.state.target_layer

    # Grad-CAM 생성
    cam_engine = GradCAM(model, target_layer)
    try:
        cam_01, class_idx = cam_engine.generate(input_tensor, class_idx=None)
    finally:
        cam_engine.close()

    pred_class = model_class.get(str(class_idx), f"class_{class_idx}")

    # 오버레이 이미지 생성/저장
    overlay = overlay_cam_on_image(orig, cam_01, alpha=0.45)
    cam_name = f"{uid}_gradcam.png"
    cam_path = os.path.join(CAM_DIR, cam_name)
    overlay.save(cam_path, format="PNG")

    # 결과 HTML: 원본 + Grad-CAM 보여주기
    html = f"""
    <html>
      <head><meta charset="utf-8"><title>Infer Result</title></head>
      <body style="font-family: Arial; padding: 24px;">
        <h2>Prediction: {pred_class} (index: {class_idx})</h2>
        <div style="display:flex; gap:24px; align-items:flex-start;">
          <div>
            <h3>Original</h3>
            <img src="/uploads/{img_name}" style="max-width:420px; border:1px solid #ddd;"/>
          </div>
          <div>
            <h3>Grad-CAM Overlay</h3>
            <img src="/gradcam/{cam_name}" style="max-width:420px; border:1px solid #ddd;"/>
          </div>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(content=html)