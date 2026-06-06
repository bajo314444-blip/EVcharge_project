# 🚀 수도권 전기차 충전소 부하 예측 및 설치 시뮬레이션 서비스 (프리미엄 관제 기능 고도화 및 UI/UX 탭 대통합) (V5.0)
> **다이내믹 요금제, TOPSIS 최적화 입지 제안, Anomaly Detection 및 AI 비서 PDF 발간 툴 탑재 (V5.0 프리미엄 스펙)**
> 
> V5.0 버전에서는 사용자 편의성과 관제 직관성을 극대화하기 위해 **4대 프리미엄 핵심 피처**를 구현하고, 기존 대시보드 레이아웃의 형태를 유지한 채 **서브 탭(`st.tabs`)으로 모듈성 있게 UI/UX를 대통합**하였습니다. 또한 Lazy Evaluation(게으른 연산)을 적용해 대용량 시뮬레이션 속도 지연 없이 "대기 시간 0초" 성능을 완전 수호했습니다.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57.0-FF4B4B.svg)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.16+-blue.svg)
![FPDF2](https://img.shields.io/badge/FPDF2-2.8.7-green.svg)
![Gemini AI](https://img.shields.io/badge/Gemini_AI-1.5_Flash-success.svg)
![Groq Llama 3](https://img.shields.io/badge/Groq-Llama_3.3_70B-orange.svg)

## 🌟 V5.0 핵심 프리미엄 관제 기능 (Highlight)
1. **💸 다이내믹 요금제 시뮬레이터 (가격 탄력성 모델)**
   - 가격 탄력성 계수 $\epsilon$ (기본값 `-0.2`)을 기반으로 피크 시간대 요금 할증 및 경부하 시간대 할인 도입 시의 시계열 수요 분산 효과를 계산하는 선형 시뮬레이션 엔진을 구축했습니다.
   - 시간대별 가격 변동률에 따른 부하 이동을 Plotly Area 차트로 실시간 비교 렌더링하고, 피크 시간대 절감률을 수치화해 줍니다.
2. **🎯 TOPSIS 다중 기준 의사결정(MCDA) 최적 입지 추천 지도**
   - 전력 부하지수, 인프라 부하지수, 충전소 밀집도 역수, 설치 비용 대비 전력망 완화율의 4대 평가 요소를 TOPSIS 알고리즘으로 종합 평가하여 신규 설치를 위한 최적 행정구역을 선발합니다.
   - 선발된 최우수 입지 TOP 5 지역에 대해 지도 상에 돋보이는 **골드 별(Star) 마커**와 상세 랭킹 카드 UI를 제공합니다.
3. **🚨 이상 징후 Anomaly Detection 관제 모듈 및 경보 배너**
   - 전압 변동성, 커넥터 온도, 패킷 유실률 등의 상태 로그를 모의 생성 및 파싱하는 Anomaly Simulator를 구축했습니다.
   - 대시보드 최상단에 위험 알림 배너를 띄우고, Leaflet CSS의 `@keyframes pulse-anim` 효과를 이용한 HTML DivIcon 기반의 **빨간색 깜빡임(Pulse) 마커**로 이상 감지 지역을 지도 상에 실시간 점멸 표시합니다.
4. **🖨️ AI 관제비서 실시간 PDF 보고서 발간 툴**
   - 특정 자치구의 현황 테이블, 증설 시뮬레이션 결과 및 다이내믹 요금제 추천 전략을 한데 묶은 3페이지 분량의 PDF 보고서 생성기(`generate_regional_report_pdf`)를 신설했습니다.
   - 사용자가 AI 관제비서에게 *"안양시 보고서 다운로드하게 해줘"*라고 지시하면, 답변창 아래에 `st.download_button`을 동적으로 렌더링해 즉각 다운로드할 수 있게 연동했습니다.
5. **🗂️ 탭 구조를 이용한 UI/UX 대통합 및 Lazy Evaluation 적용**
   - 최상단 메뉴 개수 증가로 인한 번잡함을 방지하기 위해 `🗺️ 지도 버블맵` ➔ 3개 서브 탭 분할, `💡 설치 시뮬레이션` ➔ `💡 분석 시뮬레이터` 명칭 변경 및 2개 서브 탭 분할 통합 구조를 설계했습니다.
   - 각 서브 탭 활성화 시에만 연산이 동작하도록 Lazy Evaluation을 적용하여 "대기 시간 0초"와 Rerun 억제 성능을 완성했습니다.

## 📁 V5.0 정비된 디렉토리 구조 (최종 프리미엄 아키텍처)
```text
📦 EVcharge_project
 ┣ 📂 archive/                  # 개발 히스토리 및 레거시 스크립트 백업 보관함
 ┣ 📂 components/               # UI 공통 재사용 컴포넌트
 ┣ 📂 dataset/                  # 분석용 원천 공공 데이터셋 및 위경도 파일
 ┣ 📂 results/                  # 머신러닝 학습 지표 및 종합 시각화 결과
 ┃ ┣ 📜 best_model.onnx        # Scikit-Learn 의존성 탈피를 위한 최우수 ONNX 모델
 ┃ ┗ 📜 precomputed_analytics.json # 렉 제어를 위한 대시보드 시각화 및 학술 지표 사전 연산 DB
 ┣ 📂 scripts/                  # 로컬 배치 학습 및 사전 연산 스크립트 격리 보관함
 ┣ 📂 tests/                    # 정식 회귀 테스트 및 시스템 진단 스크립트 보관함
 ┃ ┣ 📜 system_health_check.py # 대시보드 런타임 시스템 점검 도구
 ┃ ┣ 📜 test_ai_xai_dr.py      # AI 관제비서 핵심 시뮬레이션 검증 도구
 ┃ ┗ 📜 test_v48_features.py   # [★V5.0 추가] 다이내믹 요금제, TOPSIS 최적화, PDF 발간 단위 테스트
 ┣ 📂 utils/                    # 핵심 시뮬레이션 및 데이터 가공 처리 엔진 폴더
 ┃ ┣ 📜 data_processing.py     # 데이터 처리 및 ONNX/JSON 융합 로더 탑재
 ┃ ┣ 📜 models.py              # ML/DL 모델 구조 및 고정 피처 매핑 구조 적용
 ┃ ┣ 📜 optimization.py        # [★V5.0 고도화] TOPSIS 알고리즘 및 가격 탄력성 수요 분산 엔진 탑재
 ┃ ┣ 📜 pdf_generator.py       # [★V5.0 고도화] FPDF 기반 지역별 맞춤 보고서 생성기 추가
 ┃ ┣ 📜 stats_engine.py        # [★V5.0+ 신설] TOPSIS ROI 엔진 및 정책 지표 동적 연산 코어
 ┃ ┗ 📜 visualizations.py      # [★V5.0 고도화] 골드 스타 추천 마커 및 경고 Pulse 마커 적용
 ┣ 📂 views/                    # 대시보드 화면 및 라우팅 뷰
 ┃ ┣ 📜 ai_assistant.py        # [★V5.0 고도화] AI 비서 PDF 발간 툴 바인딩 및 다운로드 버튼 동적 연동
 ┃ ┣ 📜 highway_dashboard.py   # 고속도로 시뮬레이션 화면 
 ┃ ┗ 📜 urban_dashboard.py     # [★V5.0 고도화] 3원화 지도 탭, 2원화 시뮬레이션 탭 및 이상감지 배너 통합
 ┣ 📜 app.py                   # ONNX/JSON 로더 적용 최상위 라우터 및 상태 관리자
 ┣ 📜 requirements.txt         # 런타임 가벼운 배포 전용 패키지 명세서
 ┗ 📜 requirements-dev.txt     # 로컬 모델 재학습 및 배치 빌드용 전용 패키지 명세서
```

---

# 🚀 수도권 전기차 충전소 부하 예측 및 설치 시뮬레이션 서비스 (ONNX+JSON 이원화 및 지능형 관제 통합) (V4.0)
> **클라우드 서버 배포 안정성 극대화 및 패키지 역직렬화 병목 원천 차단 (V4.0 이원화 아키텍처)**
> 
> V4.0 버전에서는 로컬 환경과 배포 서버 간의 패키지 버전 불일치로 인한 역직렬화 크래시 및 메모리 부족(OOM) 문제를 완전 차단하기 위해 **ONNX + JSON 이원화(Two-track) 구동 아키텍처**를 탑재했습니다. 또한 최저 모델 및 오차 지표 정밀 매핑, **동적 Gemini 모델 스캔 및 바인딩**을 도입하여 지능형 관제 비서의 응답성 및 신뢰도를 상용 서비스 수준으로 끌어올렸습니다.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57.0-FF4B4B.svg)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.16+-blue.svg)
![skl2onnx](https://img.shields.io/badge/skl2onnx-1.15-orange.svg)
![Gemini AI](https://img.shields.io/badge/Gemini_AI-1.5_Flash-success.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-1.8.0-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-3.0.3-blue.svg)
![Random Forest](https://img.shields.io/badge/Model-Random%20Forest-lightgrey.svg)
![Gradient Boosting](https://img.shields.io/badge/Model-Gradient%20Boosting-critical.svg)
![1D-CNN](https://img.shields.io/badge/Model-1D--CNN-red.svg)
![Tabular Transformer](https://img.shields.io/badge/Model-Tabular%20Transformer-yellow.svg)

## 🌟 V4.0 핵심 구동 및 AI 고도화 기능 (Highlight)
1. **📦 ONNX 기반 초고속 예측 엔진 (`best_model.onnx`)**
   - Scikit-Learn 버전 민감도가 높은 모델 직렬화 방식(joblib/pickle)을 폐기하고, 최우수 튜닝 모델을 ONNX 형식으로 변환하여 `onnxruntime`만으로 구동되도록 가볍고 안전하게 고도화했습니다.
   - 단일/배치 데이터 입력을 통합 처리하는 `ONNXModelWrapper`를 설계하여 기존 Scikit-Learn 예측 인터페이스와 매끄럽게 호환되도록 개발했습니다.
2. **💾 시각화 및 학술 지표 JSON 데이터 직렬화 (`precomputed_analytics.json`)**
   - 차원축소(t-SNE/UMAP/CCA) 투영 좌표와 DCA, Bootstrap CI, Nested CV, 적대적 노이즈 평가, Ablation Study, Survival KM 생존곡선 등 무거운 학술 연산을 학습 완료 시점에 선처리하여 JSON 파일로 보관합니다.
   - 런타임 시 0.01초 내에 파싱 및 Pandas DataFrame 복원을 완료하여 서버의 Rerun 렉과 메모리(OOM) 병목을 소거했습니다.
3. **📐 고정 피처 구조 수립을 통한 차원 정합성 유지 (`FIXED_FEATURES`)**
   - 학습과 실시간 추론 시 데이터 피처 구조의 일관성을 확보하기 위해 13개 고정 피처 구조를 강제 적용하였으며, `make_feature_matrix`에서 `reindex`를 수행해 차원 충돌 문제를 원천 예방했습니다.
4. **🤖 AI 관제비서 (EV-Charge AI) 모델 및 데이터 바인딩 정상화**
   - **Gemini 모델 동적 스캔**: API Key별 활성화된 모델 목록(`genai.list_models()`)을 동적으로 감지하여 사용 가능한 최적 모델을 가동시켜 404 (Not Found) 에러를 방지했습니다.
   - **데이터 바인딩 정밀화**: 시스템 컨텍스트의 최적 모델명(`best_name`)과 실제 오차값(RMSE) 매핑 구조를 정상화하고, `지역` 컬럼 매핑을 수정하여 구체적인 행정구역(예: `경기 안양시 동안구`)이 AI 분석에 반영되도록 조치했습니다.
5. **📊 대화형 설명 가능한 AI (Interactive XAI Chat) 결합**
   - AI 관제비서 채팅창을 통해 특정 행정구역의 부하 예측 원인 분석을 요청하면, 백엔드에서 생성된 SHAP Local Waterfall Plot 등 XAI 시각화 자료를 Plotly 인터랙티브 차트로 즉시 채팅창 내에 동적 서빙합니다.
6. **🔋 스마트 그리드 수요반응 (DR) 및 피크 컷 시뮬레이션**
   - 단순 충전기 증설 예측을 넘어 시간대별 충전 제어 정책(V2G 및 전력 50% 분배제한) 도입 시의 전력 부하지수 분산 완화 효과를 계산하는 정책 시뮬레이터 기능을 융합했습니다.

## 📁 V4.0 정비된 디렉토리 구조 (이원화 아키텍처 탑재)
V4.0 버전에서는 ONNX 모델 및 사전 연산 DB 데이터를 관리하기 위해 `results/` 폴더 내에 `best_model.onnx`와 `precomputed_analytics.json`이 신규 도입되었습니다.

```text
📦 EVcharge_project
 ┣ 📂 archive/                  # 개발 히스토리 격리 보관함
 ┣ 📂 components/               # UI 공통 제사용 컴포넌트
 ┣ 📂 dataset/                  # 충전소 데이터 및 지오코딩 JSON 데이터
 ┣ 📂 results/                  # ML 성능 분석 정적 차트 저장소
 ┃ ┣ 📜 best_model.onnx        # [★V4.0 추가] 최우수 RandomForest ONNX 모델
 ┃ ┣ 📜 precomputed_analytics.json # [★V4.0 추가] 사전 연산 시각화/학술 지표 DB
 ┃ ┗ 📜 trained_model_state.joblib # 레거시 호환용 모델 파일
 ┣ 📂 utils/                    # 핵심 연산 및 전처리 코어 엔진 [V4.0 융합 로더 탑재]
 ┃ ┣ 📂 fonts/                  # dynamic 한글 나눔고딕 서체 폴더
 ┃ ┃ ┣ 📜 NanumGothic.ttf
 ┃ ┃ ┗ 📜 NanumGothicBold.ttf
 ┃ ┣ 📜 data_processing.py      # 데이터 처리 및 ONNX/JSON 융합 로더 추가
 ┃ ┣ 📜 models.py               # ML/DL 회귀 모델 구축 및 고정 피처 구조 적용
 ┃ ┣ 📜 optimization.py         # 최적 입지 LP 및 모델 강건성/DCA/생존분석 알고리즘
 ┃ ┣ 📜 pdf_generator.py        # PDF 자동 생성 모듈 (Matplotlib Headless 지원)
 ┃ ┣ 📜 visualizations.py      # Folium 지도 인터랙션 및 Plotly 에지 시각화
 ┃ ┗ 📜 __init__.py
 ┣ 📂 views/                    # 프론트엔드 라우팅 뷰
 ┃ ┣ 📜 highway_dashboard.py    # 고속도로 시뮬레이션 화면 
 ┃ ┗ 📜 urban_dashboard.py      # 도심 분석 대시보드 및 precomputed ROC 렌더러 연동
 ┣ 📜 app.py                    # ONNX/JSON 로더 적용 최상위 라우터
 ┣ 📜 requirements.txt          # 패키지 의존성 파일 (onnx, onnxruntime, skl2onnx 추가)
 ┗ 📜 README.md                 # 프로젝트 명세서 (본 파일)
```

---

# 🚀 수도권 전기차 충전소 부하 예측 및 설치 시뮬레이션 서비스 (스마트 UX 성능 최적화 및 Clean Architecture) (V3.0)
> **클라우드 서버(Streamlit Cloud) 배포 안정성 및 반응 속도 10배 고속화 (V3.0 스마트 UX 패키지)**
> 
> V3.0 버전에서는 1GB RAM 및 1 vCPU의 제한적인 클라우드 서버 환경에서도 100% 동적 예측 모델 피팅 및 양방향 지도 이벤트 시뮬레이션을 원활하게 수행할 수 있도록 **게으른 연산(Lazy Evaluation)**, **지연 로딩(Lazy Fitting)**, **Folium 렌더링 락(Rerun Lock)**, **Headless 차트 생성기** 및 **경로 독립형 폰트 바인딩**을 도입하여 최고의 안정성과 상용 서비스 수준의 성능을 구현했습니다.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57.0-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-1.8.0-orange.svg)
![Matplotlib Headless](https://img.shields.io/badge/Matplotlib_Headless-3.10.9-blue.svg)
![FPDF2](https://img.shields.io/badge/FPDF2-2.8.7-green.svg)
![NetworkX](https://img.shields.io/badge/NetworkX-3.6.1-orange.svg)
![SHAP](https://img.shields.io/badge/SHAP-0.52.0-blue.svg)
![UMAP](https://img.shields.io/badge/UMAP-0.5.12-blue.svg)
![Random Forest](https://img.shields.io/badge/Model-Random%20Forest-lightgrey.svg)
![Gradient Boosting](https://img.shields.io/badge/Model-Gradient%20Boosting-critical.svg)
![1D-CNN](https://img.shields.io/badge/Model-1D--CNN-red.svg)
![Tabular Transformer](https://img.shields.io/badge/Model-Tabular%20Transformer-yellow.svg)

## 🌟 V3.0 스마트 UX & 배포 최적화 기능 (Highlight)
1. **🦥 게으른 연산 (Lazy Evaluation) & 세그먼트 메뉴 개편**
   - 기존의 무거운 `st.tabs` 렌더링 방식을 탈피하고 가로형 세그먼트 라디오 네비게이션 구조로 전면 혁신했습니다.
   - 비활성화된 메뉴 내부의 무거운 머신러닝 연산(t-SNE, UMAP, SHAP 등)이 실행 계통에서 100% 제외(Lazy)되도록 설계하여 페이지 Rerun 시 연산 지연 속도를 10배 이상 고속화했습니다.
2. **⏳ SMOTE 지연 로딩 (Lazy Fitting) 및 UX 스피너 적용**
   - 최초 앱 진입 시 SMOTE 학습 모델 14개를 전부 학습하던 병목을 해결하고, 사용자가 `📊 예측 모델 비교` 탭에 접근한 시점에 1회 비동기로 학습 및 캐싱을 실행하도록 지연 로딩(Lazy Fitting)을 구현하여 초기 구동 대기 시간을 50% 단축했습니다.
3. **🗺️ Folium 양방향 이벤트 미세 제어 (Rerun Lock)**
   - 지도를 확대하거나 드래그할 때 웹페이지 전체가 지속해서 다시 그려지며 먹통이 되던 문제를 해소하기 위해 `st_folium`에 `returned_objects=["last_object_clicked"]`를 지정해 **"충전소 버블을 클릭했을 때만"** Rerun이 유발되도록 최적화했습니다.
4. **📊 Kaleido 비의존성 Headless 이미지 생성 엔진 도입**
   - Linux 및 Chrome 헤드리스 드라이버가 누락된 클라우드 배포 서버 환경에서 Plotly를 PNG로 내보낼 시 발생하는 `Kaleido RuntimeError`를 원천 예방하기 위해, **Matplotlib 백엔드 렌더러**를 구현하여 RAM 내 가상 파일(BytesIO) 형태로 정적 플롯을 안전하게 생성, FPDF PDF 리포트에 깨짐 없이 동적 각인합니다.
5. **🔤 경로 독립형 나눔고딕 폰트 dynamic 바인딩**
   - OS(Windows/Linux) 및 설치 환경에 따른 나눔고딕 폰트의 `FileNotFoundError` 문제를 해결하기 위해, `utils/fonts/` 전용 폰트 디렉토리를 구축하고 dynamic 상대 경로를 통해 동적 서체 로드를 바인딩하여 무결성 다중 폰트 한글 출력을 수호합니다.

## 📁 V3.0 정비된 디렉토리 구조 (스마트 UX & dynamic 폰트 탑재)
V3.0 버전에서는 dynamic 폰트 바인딩을 위한 전용 폰트 디렉토리(`utils/fonts/`)를 추가 구축하여 클라우드 서버 환경에서의 폰트 호환성 문제를 완벽히 해결했습니다.

```text
📦 EVcharge_project
 ┣ 📂 archive/                  # 개발 히스토리 격리 보관함
 ┃ ┣ 📜 debug_shap.py           # SHAP 설명력 설명 디버깅 소스
 ┃ ┗ 📜 refactor_script.py      # 리팩토링용 보조 소스
 ┣ 📂 components/               # UI 공통 제사용 컴포넌트
 ┃ ┗ 📜 sidebar.py              # 멀티뷰 제어 사이드바 컨트롤러
 ┣ 📂 dataset/                  # 충전소 데이터 및 지오코딩 JSON 데이터
 ┣ 📂 results/                  # ML 성능 분석 정적 차트 저장소
 ┣ 📂 utils/                    # 핵심 연산 및 전처리 코어 엔진 [V3.0 폰트 최적화]
 ┃ ┣ 📂 fonts/                  # [★V3.0 추가] dynamic 한글 나눔고딕 서체 폴더
 ┃ ┃ ┣ 📜 NanumGothic.ttf
 ┃ ┃ ┗ 📜 NanumGothicBold.ttf
 ┃ ┣ 📜 data_processing.py      # 데이터 처리 파이프라인 및 st.cache 캐싱 레이어 
 ┃ ┣ 📜 models.py               # ML/DL 회귀 모델 구축 및 훈련 로직
 ┃ ┣ 📜 optimization.py         # 최적 입지 LP 및 모델 강건성/DCA/생존분석 알고리즘
 ┃ ┣ 📜 pdf_generator.py        # FPDF 기반 PDF 국가 보고서 자동 생성 레이아웃 (Matplotlib Headless 지원)
 ┃ ┣ 📜 visualizations.py      # Folium 지도 인터랙션 및 Plotly 에지 시각화
 ┃ ┗ 📜 __init__.py
 ┣ 📂 views/                    # 프론트엔드 라우팅 뷰
 ┃ ┣ 📜 highway_dashboard.py    # 고속도로 시뮬레이션 화면 
 ┃ ┗ 📜 urban_dashboard.py      # 도심 분석 대시보드 + 게으른 연산 및 st_folium 최적화 [★V3.0 튜닝 완료]
 ┣ 📜 app.py                    # 최상위 라이트웨이트 라우터 & 상태 관리자 [★V3.0 지연학습 적용]
 ┣ 📜 requirements.txt          # 패키지 의존성 파일
 ┗ 📜 README.md                 # 프로젝트 명세서 (본 파일)
```

---

# 🚀 수도권 전기차 충전소 부하 예측 및 설치 시뮬레이션 서비스 (대통합 아키텍처 및 고성능 시뮬레이터) (V2.0)
> **Clean Architecture 2단계 기반의 모듈화 고도화 및 학술적 검증 엔진 탑재**
> 
> V2.0 버전에서는 물리적 파일 구조를 도메인별 응집력에 따라 축소 단축하고, 고속도로망 최적화 및 8대 학술적 모델 강건성 시뮬레이터를 전면 탑재한 상용 서비스 수준의 웹 관제 솔루션을 제공합니다.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57.0-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-1.8.0-orange.svg)
![Kaleido](https://img.shields.io/badge/Kaleido-1.3.0-blue.svg)
![FPDF2](https://img.shields.io/badge/FPDF2-2.8.7-green.svg)
![NetworkX](https://img.shields.io/badge/NetworkX-3.6.1-orange.svg)
![SciPy](https://img.shields.io/badge/SciPy-1.17.1-success.svg)
![Random Forest](https://img.shields.io/badge/Model-Random%20Forest-lightgrey.svg)
![Gradient Boosting](https://img.shields.io/badge/Model-Gradient%20Boosting-critical.svg)
![1D-CNN](https://img.shields.io/badge/Model-1D--CNN-red.svg)
![Tabular Transformer](https://img.shields.io/badge/Model-Tabular%20Transformer-yellow.svg)

## 🌟 V2.0 핵심 업그레이드 기능 (Highlight)
1. **🛣️ 고속도로망 최적 입지 제안 (Linear Programming)**
   - 수도권 주요 고속도로 IC/휴게소 데이터를 공간 조인(BallTree)하여 충전소 용량을 산출하고, 선형 계획법(linprog)을 이용해 한정된 예산 내에서 최대 효율을 내는 최적 입지 대수를 노드별로 제안합니다.
   - **상하행선 분리 네트워크 에지 맵**: 명절/출퇴근 혼잡 시나리오별 실시간 교통량 부하를 에지(Edge) 두께와 색상 지도로 가독성 높게 표출합니다.
2. **📋 국가 공무 서식 PDF 리포트 엔진**
   - FPDF 엔진 및 Plotly 스냅샷 내보내기 도구(`kaleido`)를 융합하여, 원클릭으로 TOP 3 취약지역 분석 및 주요 Feature Importance 바 차트가 깨짐 없이 깔끔하게 각인된 **정부/지자체 보고 양식의 PDF 리포트**를 발행합니다.
3. **🛡️ 8대 학술적 모델 검증 및 강건성 평가**
   - **부트스트랩 95% 신뢰구간 (Bootstrap CI)** 산출
   - **중첩 10-겹 교차검증 (Nested 10-fold CV)을** 통한 과적합 제로 성능 검증
   - **공간적 외부 검증 (Spatial External Validation)으로** 미학습 새로운 구(서울/경기/인천 Holdout) 예측 성능 비교
   - **적대적 노이즈 방어력 평가:** 가우시안 노이즈 주입 시의 강건도 평가
   - **피처 중요도 순차 제거 분석 (Ablation Study)** 민감도 플롯
   - **의사결정 곡선 분석 (Decision Curve Analysis - DCA)** 순수 혜택(Net Benefit) 평가
   - **생존 분석 시뮬레이터:** 카플란-마이어(Kaplan-Meier) 곡선 기반 과부하 도달 시간(Time-to-Overload) 시뮬레이션

## 📁 V2.0 정비된 디렉토리 구조 (Clean Architecture 2.0)
V2.0에서는 불필요한 보조 스크립트와 임시 파일을 과감히 정리하고, 다음과 같은 고결합도/고응집성 파일 배치 구조를 구축했습니다.

```text
📦 EVcharge_project
 ┣ 📂 archive/                  # [V2.0] 개발 히스토리 격리 보관함
 ┃ ┣ 📜 debug_shap.py           # SHAP 설명력 디버깅 소스
 ┃ ┗ 📜 refactor_script.py      # 리팩토링용 보조 소스
 ┣ 📂 components/               # UI 공통 제사용 컴포넌트
 ┃ ┗ 📜 sidebar.py              # 멀티뷰 제어 사이드바 컨트롤러
 ┣ 📂 dataset/                  # 충전소 데이터 및 지오코딩 JSON 데이터
 ┣ 📂 results/                  # ML 성능 분석 정적 차트 저장소 (01~09번)
 ┣ 📂 utils/                    # 핵심 연산 및 전처리 코어 엔진 [통합 완료]
 ┃ ┣ 📜 data_processing.py      # 데이터 처리 파이프라인 및 st.cache 캐싱 레이어 
 ┃ ┣ 📜 models.py               # ML/DL 회귀 모델 구축, 특징 행렬 생성 및 훈련 로직
 ┃ ┣ 📜 optimization.py         # 최적 입지 LP 및 모델 강건성/생존분석 알고리즘
 ┃ ┣ 📜 pdf_generator.py        # FPDF 기반 PDF 국가 보고서 자동 생성 레이아웃
 ┃ ┣ 📜 visualizations.py      # Folium 지도 인터랙션 및 Plotly 에지 시각화
 ┃ ┗ 📜 __init__.py
 ┣ 📂 views/                    # 프론트엔드 라우팅 뷰 [통합 완료]
 ┃ ┣ 📜 highway_dashboard.py    # 고속도로 시뮬레이션 화면 
 ┃ ┗ 📜 urban_dashboard.py      # 도심 분석 대시보드 및 리포트 탭 분할 통합 화면
 ┣ 📜 app.py                    # 최상위 라이트웨이트 라우터 & 상태 관리자
 ┣ 📜 requirements.txt          # 패키지 의존성 파일
 ┗ 📜 README.md                 # 프로젝트 명세서 (본 파일)
```

## 🚀 V2.0 추가 의존성 설치 및 구동
정부 양식의 고품질 PDF 리포트 출력 및 네트워크 그래프, UMAP 등의 시각화 모듈을 완전히 활용하기 위해 다음 의존성이 추가 정의되었습니다.

```bash
# 1. 툴킷 및 고급 패키지 포함 일괄 설치
pip install -r requirements.txt

# 2. 애플리케이션 시작
streamlit run app.py
```

## 🛠️ V2.0 확장 기술 스택 (Extended Tech Stack)
*   **Optimization Solver**: `SciPy` (linprog Highs-Solver)
*   **Network Interaction**: `NetworkX` (상관 관계 및 에지 연결망 분석)
*   **Image Serialization**: `Kaleido` (Plotly 벡터 차트 고속 PNG 각인 엔진)
*   **Report Exporter**: `FPDF2` (다중 폰트 및 그리드 시스템 지원 PDF 생성기)
*   **Dimension Reduction**: `UMAP-learn`, `TSNE`, `PCA`

---

# ⚡ 수도권 전기차 충전소 부하 예측 및 설치 시뮬레이션 서비스 (V1.0)
> **Seoul Metropolitan Area EV Charging Station Load Prediction & Simulation**
> 
> 공공데이터를 활용하여 수도권(서울, 경기, 인천) 전기차 충전 인프라의 과부하 위험 지역을 예측하고, 신규 충전기 설치 시나리오를 시뮬레이션할 수 있는 Streamlit 기반의 인터랙티브 웹 서비스입니다.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57.0-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-1.8.0-orange.svg)
![Numpy Custom](https://img.shields.io/badge/Numpy_Custom-2.4.6-success.svg)
![Plotly](https://img.shields.io/badge/Plotly-6.7.0-blue.svg)
![Folium](https://img.shields.io/badge/Folium-0.20.0-green.svg)
![Random Forest](https://img.shields.io/badge/Model-Random%20Forest-lightgrey.svg)
![Gradient Boosting](https://img.shields.io/badge/Model-Gradient%20Boosting-critical.svg)
![1D-CNN](https://img.shields.io/badge/Model-1D--CNN-red.svg)
![Tabular Transformer](https://img.shields.io/badge/Model-Tabular%20Transformer-yellow.svg)

## 📌 주요 기능 (Features)
- **🗺️ 현재 부하 버블맵**: 시도/시군구별 전력 부하지수 및 인프라 지수를 Folium 지도 위에 시각화 (자가용/사업자용 분리 조회 가능)
- **📈 월별 부하 추이 분석**: 환경부 공공급속 충전기 연월별 충전량 및 충전 횟수 시계열 추이 분석
- **🔮 예측 모델 성능 비교**: 5개의 트리 기반 머신러닝(RandomForest, GBM 등) 및 2개의 커스텀 딥러닝(1D-CNN, Tabular Transformer) 모델 성능 테스트 및 비교
- **💡 설치 시뮬레이션**: 특정 지역에 신규 충전기(ex. 100kW 10대) 추가 시 감소하는 전력 부하지수 효과 계산
- **🧠 SHAP & LIME XAI**: 모델이 왜 특정 지역을 고부하 지역으로 예측했는지 설명 가능한 인공지능(Explainable AI) 시각화 제공

## 📁 디렉토리 구조 (Directory Structure)
유지보수 및 확장을 위해 메인 웹 뷰와 데이터/모델 로직을 분리하여 설계되었습니다.

```text
📦 DataAnalysis_3-1 
┣ 📂 dataset/ # 공공데이터 원본 파일 (충전소 위치, 등록대수, 한전 전력판매량 등) 
┣ 📂 utils/ # 핵심 서비스 비즈니스 로직 모듈 
┃ ┣ 📜 data_processing.py # 다중 인코딩 지원 데이터 병합 및 전처리 파이프라인 
┃ ┣ 📜 models.py # ML/DL 회귀 모델 구축, 특징 행렬 생성 및 훈련 로직 
┃ ┗ 📜 visualizations.py # Folium 지도, SHAP, TableOne 등 렌더링 로직 
┣ 📜 app.py # Streamlit 웹 어플리케이션 메인 엔트리 포인트 
┣ 📜 requirements.txt # 패키지 의존성 명세서 
┗ 📜 .gitignore # 깃허브 제외 파일 목록
```

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
