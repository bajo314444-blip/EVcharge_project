# ⚡ 수도권 전기차 충전소 부하 예측 및 설치 시뮬레이션 서비스
> **Seoul Metropolitan Area EV Charging Station Load Prediction & Simulation**
> 
> 공공데이터를 활용하여 수도권(서울, 경기, 인천) 전기차 충전 인프라의 과부하 위험 지역을 예측하고, 신규 충전기 설치 시나리오를 시뮬레이션할 수 있는 Streamlit 기반의 인터랙티브 웹 서비스입니다.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange.svg)

## 📌 주요 기능 (Features)
- **🗺️ 현재 부하 버블맵**: 시도/시군구별 전력 부하지수 및 인프라 지수를 Folium 지도 위에 시각화 (자가용/사업자용 분리 조회 가능)
- **📈 월별 부하 추이 분석**: 환경부 공공급속 충전기 연월별 충전량 및 충전 횟수 시계열 추이 분석
- **🔮 예측 모델 성능 비교**: 5개의 트리 기반 머신러닝(RandomForest, GBM 등) 및 2개의 커스텀 딥러닝(1D-CNN, Tabular Transformer) 모델 성능 테스트 및 비교
- **💡 설치 시뮬레이션**: 특정 지역에 신규 충전기(ex. 100kW 10대) 추가 시 감소하는 전력 부하지수 효과 계산
- **🧠 SHAP & LIME XAI**: 모델이 왜 특정 지역을 고부하 지역으로 예측했는지 설명 가능한 인공지능(Explainable AI) 시각화 제공

## 📁 디렉토리 구조 (Directory Structure)
유지보수 및 확장을 위해 메인 웹 뷰와 데이터/모델 로직을 분리하여 설계되었습니다.

📦 DataAnalysis_3-1
 ┣ 📂 dataset/               # 공공데이터 원본 파일 (충전소 위치, 등록대수, 한전 전력판매량 등)
 ┣ 📂 utils/                 # 핵심 서비스 비즈니스 로직 모듈
 ┃ ┣ 📜 data_processing.py   # 다중 인코딩 지원 데이터 병합 및 전처리 파이프라인
 ┃ ┣ 📜 models.py            # ML/DL 회귀 모델 구축, 특징 행렬 생성 및 훈련 로직
 ┃ ┗ 📜 visualizations.py    # Folium 지도, SHAP, TableOne 등 렌더링 로직
 ┣ 📜 app.py                 # Streamlit 웹 어플리케이션 메인 엔트리 포인트
 ┣ 📜 requirements.txt       # 패키지 의존성 명세서
 ┗ 📜 .gitignore             # 깃허브 제외 파일 목록

## 📊 사용된 데이터셋 (Datasets)
본 프로젝트는 다음의 최신 공공데이터들을 융합하여 분석했습니다.
- 한국교통안전공단 - 전국 전기차 차종별/용도별 차량 등록대수
- 한국전력공사 - 전기차 충전소 위치/경도 및 시간대별 충전부하
- 한국전력공사 - 충전소별 충전기 용량 정보 및 전국 전력 판매실적
- 환경부 - 공공급속 충전기 연월별 충전량 및 충전시간 이력
- 서울/경기 지자체 - 시도별 전기차 충전소 설치 현황

## 🚀 실행 방법 (How to Run)

1. 저장소를 클론합니다.
```bash
git clone https://github.com/본인계정/저장소이름.git
cd 저장소이름
```

2. 패키지를 설치합니다.
```bash
pip install -r requirements.txt
```

3. Streamlit 앱을 실행합니다.
```bash
streamlit run app.py
```

## 🛠️ 기술 스택 (Tech Stack)
- **Data Preprocessing**: `Pandas`, `Numpy`
- **Machine Learning**: `Scikit-learn` (RandomForest, ExtraTrees, GBM, KNN, CCA, t-SNE)
- **Deep Learning**: Numpy Custom Implementation (1D-CNN, Tabular Transformer)
- **Visualization**: `Plotly`, `Folium`, `SHAP`
- **Web Framework**: `Streamlit`
