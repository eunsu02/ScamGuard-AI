import torch
from transformers import BertTokenizer, BertForSequenceClassification

MY_MODEL_PATH = "./models/kobert_model"

# 모델과 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained(MY_MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MY_MODEL_PATH)
model.eval()

def predict_scam_kobert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        # 사기(Label 1)일 확률 계산
        prob = torch.softmax(outputs.logits, dim=1)[0][1].item()

    return prob