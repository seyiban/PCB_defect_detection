# 윤효열
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

def predict(img, model, threshold=0.5):
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]
        prob_defect = probs[0].item()
    pred_class = 1 if prob_defect >= threshold else 0
    confidence = prob_defect if pred_class == 1 else 1 - prob_defect
    return pred_class, confidence, prob_defect

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
        padding: 24px 28px; border-radius: 14px; margin: 16px 0;
    }
    .result-normal {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        border-left: 6px solid #16a34a;
        padding: 24px 28px; border-radius: 14px; margin: 16px 0;
    }
    .result-title { font-size: 26px; font-weight: bold; margin: 0 0 6px 0; }
    .result-sub { font-size: 15px; margin: 0; opacity: 0.85; }
    .metric-box {
        background: white; border-radius: 14px; padding: 20px 16px;
        text-align: center; box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border: 1px solid #f1f5f9;
    }
    .metric-value { font-size: 30px; font-weight: bold; color: #1e293b; }
    .metric-label { font-size: 12px; color: #94a3b8; margin-top: 6px; letter-spacing: 0.05em; text-transform: uppercase; }
    .info-box {
        background: #eff6ff; border: 1px solid #bfdbfe;
        border-radius: 12px; padding: 16px 20px;
        font-size: 14px; color: #1e40af; line-height: 1.6;
    }
    .section-title {
        font-size: 16px; font-weight: 700; color: #475569;
        margin: 24px 0 12px 0; letter-spacing: 0.03em; text-transform: uppercase;
    }
    .img-card {
        background: white; border-radius: 14px; padding: 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07); border: 1px solid #f1f5f9;
        text-align: center;
    }
    .img-label { font-size: 13px; color: #64748b; margin-top: 10px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🔍 PCB 결함 탐지 시스템")
st.markdown("DenseNet-121 기반 PCB 이미지 결함 자동 검사 시스템")
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
| 에폭 | 50 |
| 학습률 | 0.0001 |
| Threshold | 0.5 |
""")
    st.divider()
    st.markdown("## 💡 사용 가이드")
    st.markdown("""
1. PCB 이미지를 업로드하세요
2. AI가 자동으로 결함을 분석합니다
3. 결과 및 확률을 확인하세요
4. 필요시 이미지를 다운로드하세요

**지원 형식**: JPG, JPEG, PNG
""")

uploaded_file = st.file_uploader("📁  PCB 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.markdown("""
<div class="info-box">
ℹ️ <b>PCB 이미지를 업로드하면 자동으로 결함 분석을 시작합니다.</b><br>
정상 및 결함 PCB 이미지를 업로드해 AI 판정 결과와 확률을 확인할 수 있습니다.
</div>
""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-title">📈 모델 성능 요약</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-box"><div class="metric-value">50</div><div class="metric-label">학습 에폭</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-box"><div class="metric-value">1e-4</div><div class="metric-label">Learning Rate</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-box"><div class="metric-value">PCB2+3</div><div class="metric-label">학습 데이터</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-box"><div class="metric-value">0.5</div><div class="metric-label">기본 Threshold</div></div>', unsafe_allow_html=True)

else:
    with st.spinner("🔄  AI가 이미지를 분석 중입니다..."):
        try:
            model = load_model()
            img = Image.open(uploaded_file).convert('RGB')
        except Exception as e:
            st.error(f"❌  이미지를 불러오는 데 실패했습니다: {e}")
            st.stop()

        img_resized = img.resize((224, 224))

        try:
            pred_class, confidence, prob_defect = predict(img, model, threshold)
        except Exception as e:
            st.error(f"❌  분석 중 오류가 발생했습니다: {e}")
            st.stop()

    if pred_class == 1:
        st.markdown(f"""
<div class="result-defect">
    <p class="result-title">⚠️ 결함 감지 (Defect Detected)</p>
    <p class="result-sub" style="color:#991b1b;">결함 확률 {prob_defect*100:.1f}% — 즉각적인 검토가 필요합니다</p>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div class="result-normal">
    <p class="result-title">✅ 정상 (Normal)</p>
    <p class="result-sub" style="color:#14532d;">정상 확률 {(1-prob_defect)*100:.1f}% — 이상이 감지되지 않았습니다</p>
</div>""", unsafe_allow_html=True)

    st.markdown("")
    img_col, stat_col = st.columns([1, 1.6], gap="large")

    with img_col:
        st.markdown('<div class="section-title">📸 입력 이미지</div>', unsafe_allow_html=True)
        st.markdown('<div class="img-card">', unsafe_allow_html=True)
        st.image(img_resized, use_container_width=True)
        st.markdown(f'<div class="img-label">{uploaded_file.name}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with stat_col:
        st.markdown('<div class="section-title">📊 분석 결과</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            color = "#dc2626" if pred_class == 1 else "#16a34a"
            label = "결함" if pred_class == 1 else "정상"
            st.markdown(f'<div class="metric-box"><div class="metric-value" style="color:{color}">{label}</div><div class="metric-label">판정 결과</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{threshold}</div><div class="metric-label">적용 임계값</div></div>', unsafe_allow_html=True)

        st.markdown("")

        st.markdown("**결함 확률**")
        st.progress(prob_defect, text=f"{prob_defect*100:.1f}%")

        st.markdown("**정상 확률**")
        st.progress(1 - prob_defect, text=f"{(1-prob_defect)*100:.1f}%")

        st.markdown("**판정 신뢰도**")
        st.progress(confidence, text=f"{confidence*100:.1f}%")

        st.markdown("")
        st.download_button(
            label="📥 이미지 다운로드",
            data=pil_to_bytes(img_resized),
            file_name="pcb_result.png",
            mime="image/png",
            use_container_width=True
        )
