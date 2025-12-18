import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertModel
from safetensors.torch import load_file

MODEL_PATH = "/content/drive/MyDrive/CAPSTONE PROJECT/MODELS/phishbert_role_v1"

# -----------------------------
# Tokenizer
# -----------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# -----------------------------
# Role-aware model
# -----------------------------
class PhishBertWithRole(nn.Module):
    def __init__(self, num_roles=16, role_dim=32):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.role_embed = nn.Embedding(num_roles, role_dim)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768 + role_dim, 2)

    def forward(self, input_ids, attention_mask, role_id):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        role_vec = self.role_embed(role_id)
        fused = torch.cat([cls, role_vec], dim=1)
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return logits

# -----------------------------
# Load model
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PhishBertWithRole(num_roles=16, role_dim=32)
state = load_file(f"{MODEL_PATH}/model.safetensors")
model.load_state_dict(state)
model.to(device)
model.eval()

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="PhishBERT Role-Aware API")

class PredictRequest(BaseModel):
    email_text: str
    role_id: int

@app.post("/predict")
def predict(req: PredictRequest):
    inputs = tokenizer(
        req.email_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    role_tensor = torch.tensor([req.role_id]).to(device)

    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"], role_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return {
        "label": "Phishing" if pred == 1 else "Legitimate",
        "confidence": round(confidence, 4),
        "role_id": req.role_id
    }

@app.get("/")
def root():
    return {"status": "PhishBERT Role-Aware API is running"}
