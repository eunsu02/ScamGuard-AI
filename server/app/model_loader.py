import torch
import timm


import torch
import timm
from torchvision import transforms
from PIL import Image
import os
import torch
import timm
from torchvision import transforms
from PIL import Image


# M2 맥북 가속 설정
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 💡 [수정] 현재 파일의 위치를 기준으로 모델 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "scamguard_model.pth")

# 모델 구조 정의
model = timm.create_model(
    "legacy_xception", num_classes=2
)  # 경고 메시지에 따라 legacy_xception 권장
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

# 가중치 로드
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# 전처리 도구
transformer = transforms.Compose(
    [
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


def predict_deepfake(face_img):
    """
    main.py에서 호출할 실제 추론 함수
    """
    if isinstance(face_img, Image.Image):
        image = face_img
    else:
        image = Image.fromarray(face_img)

    input_tensor = transformer(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.nn.functional.softmax(output, dim=1)[0][1].item()

    return prob


def get_model(model_path: str):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = timm.create_model("xception", num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device
