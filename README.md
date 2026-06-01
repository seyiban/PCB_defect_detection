# 🔍 PCB 결함 탐지 시스템

DenseNet-121 기반 PCB 이미지 결함 자동 탐지 및 분류 시스템

---

## 📌 프로젝트 개요

VisA 데이터셋의 PCB2/PCB3 이미지를 활용해 정상/결함을 분류하는 딥러닝 모델을 학습하고,
Streamlit 웹앱으로 배포한 PCB 결함 탐지 시스템입니다.

---

## 🗂️ 데이터셋

- **출처**: [VisA (Visual Anomaly)](https://github.com/amazon-science/spot-difference)
- **사용 카테고리**: PCB2, PCB3
- **클래스**: Normal / Anomaly
- **학습 데이터**: train 각 700장, val/test 각 150장 (normal 기준)

---

## 🧠 모델 구조

| 항목 | 내용 |
|------|------|
| 베이스 모델 | DenseNet-121 (ImageNet pretrained) |
| 출력층 | Linear(1024 → 2) |
| 손실 함수 | CrossEntropyLoss (anomaly 3배 가중치) |
| 옵티마이저 | Adam (레이어별 lr 차등 적용) |
| 스케줄러 | ReduceLROnPlateau |
| Best 기준 | Anomaly Recall |

---

## ⚙️ 학습 설정

| 항목 | 값 |
|------|-----|
| Epochs | 50 |
| Batch size | 32 |
| Learning rate | 0.0001 |
| Image size | 224×224 |
| Best epoch | 26 |

---

## 🚀 실행 방법

### 로컬 실행

1. 패키지 설치
```bash
pip install -r requirements.txt
```

2. 모델 파일 준비
`best_model_densenet.pth` 파일을 프로젝트 루트에 위치시킵니다.

3. 앱 실행
```bash
streamlit run app.py
```

---

## 📁 파일 구조

PCB_defect_detection/
├── app.py                    # Streamlit 웹앱
├── best_model_densenet.pth   # 학습된 모델 가중치
├── requirements.txt          # 패키지 목록
└── README.md                 # 프로젝트 설명

---

## 🖥️ 웹앱 기능

- PCB 이미지 업로드
- 정상/결함 자동 분류
- 결함 확률 및 정상 확률 시각화
- 결함 판정 임계값 조절 (사이드바)
- 결과 이미지 다운로드