import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import io

@st.cache_resource
def load_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    model.load_state_dict(torch.load('best_model_densenet.pth', map_location='cpu'))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def pil_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

st.set_page_config(page_title="PCB 결함 탐지 시스템", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    .result-defect {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border-left: 6px solid #dc2626;
        padding: 20px; border-radius: 12px; margin: 10px 0;
    }
    .result-normal {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        border-left: 6px solid #16a34a;
        padding: 20px; border-radius: 12px; margin: 10px 0;
    }
    .result-text { font-size: 28px; font-weight: bold; margin: 0; }
    .metric-box {
        background: white; border-radius: 12px; padding: 16px;
        text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .metric-value { font-size: 32px; font-weight: bold; color: #1e293b; }
    .metric-label { font-size: 13px; color: #64748b; margin-top: 4px; }
    .info-box {
        background: #eff6ff; border: 1px solid #bfdbfe;
        border-radius: 10px; padding: 14px;
        font-size: 13px; color: #1e40af;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🔍 PCB 결함 탐지 시스템")
st.markdown("AI 기반 PCB 이미지 결함 자동 검사")
st.divider()

with st.sidebar:
    st.markdown("## ⚙️ 검사 설정")
    threshold = st.slider("결함 판정 임계값", 0.1, 0.9, 0.5, 0.05,
                          help="값이 낮을수록 결함을 더 민감하게 탐지합니다")
    st.divider()
    st.markdown("## 📊 모델 정보")
    st.markdown("""
| 항목 | 내용 |
|------|------|
| 모델 | DenseNet-121 |
| 데이터 | VisA PCB2/PCB3 |
| Accuracy | 98.79% |
| 결함 Recall | 86.7% |
| Threshold | 0.5 |
""")
    st.divider()
    st.markdown("## 💡 사용 가이드")
    st.markdown("""
1. PCB 이미지를 업로드하세요
2. AI가 자동으로 결함을 분석합니다
3. 판정 결과와 신뢰도를 확인하세요

**지원 형식**: JPG, JPEG, PNG
""")

uploaded_file = st.file_uploader("📁  PCB 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.markdown("""
<div class="info-box">
ℹ️ <b>PCB 이미지를 업로드하면 자동으로 결함 분석을 시작합니다.</b>
</div>
""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-box"><div class="metric-value">98.79%</div><div class="metric-label">Test Accuracy</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-box"><div class="metric-value">86.7%</div><div class="metric-label">결함 Recall</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-box"><div class="metric-value">2207</div><div class="metric-label">학습 이미지 수</div></div>', unsafe_allow_html=True)

else:
    with st.spinner("🔄  AI가 이미지를 분석 중입니다..."):
        try:
            model = load_model()
            img = Image.open(uploaded_file).convert('RGB')
        except Exception as e:
            st.error(f"❌  이미지를 불러오는 데 실패했습니다: {e}")
            st.stop()

        img_resized = img.resize((224, 224))
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            prob_defect = torch.softmax(output, dim=1)[0][1].item()

        pred_class = 1 if prob_defect >= threshold else 0
        confidence = prob_defect if pred_class == 1 else 1 - prob_defect

    if pred_class == 1:
        st.markdown(f"""
<div class="result-defect">
    <p class="result-text">⚠️ 결함 감지 (Defect Detected)</p>
    <p style="margin:6px 0 0 0; color:#991b1b;">결함 확률 {prob_defect*100:.1f}% — 즉각적인 검토가 필요합니다</p>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div class="result-normal">
    <p class="result-text">✅ 정상 (Normal)</p>
    <p style="margin:6px 0 0 0; color:#14532d;">정상 확률 {(1-prob_defect)*100:.1f}% — 이상 없음</p>
</div>""", unsafe_allow_html=True)

    st.markdown("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-box"><div class="metric-value" style="color:{"#dc2626" if pred_class==1 else "#16a34a"}">{"결함" if pred_class==1 else "정상"}</div><div class="metric-label">판정 결과</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{prob_defect*100:.1f}%</div><div class="metric-label">결함 확률</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{confidence*100:.1f}%</div><div class="metric-label">판정 신뢰도</div></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown("### 📸 업로드된 이미지")
    st.image(img_resized, caption="원본 이미지", width=300)

    st.markdown("")
    st.download_button(
        label="📥 이미지 다운로드",
        data=pil_to_bytes(img_resized),
        file_name="original.png",
        mime="image/png"
    )
