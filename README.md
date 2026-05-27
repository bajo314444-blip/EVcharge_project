# 수도권 전기차 충전소 부하 예측 웹서비스

이 프로젝트는 기존 `EVcar_charge_analysis_study.ipynb`의 전처리와 모델링 흐름을 Streamlit 웹서비스로 재구성한 과제용 앱입니다.

## 실행 방법

```powershell
streamlit run app.py
```

앱 왼쪽 사이드바의 데이터 폴더 기본값은 아래 경로입니다.

```text
D:\Users\bajo3\OneDrive\바탕 화면\데분응\메일 엑셀파일
```

## 포함 기능

- 수도권 전기차 충전소 부하 버블맵
- 자가용/사업자용 그룹 비교
- 환경부 공공급속 충전기 연월별 충전량 파일 기반 월별 부하 변화
- 신규 충전기 설치 시 부하 감소 시뮬레이션
- 5개 이상 머신러닝 모델과 2개 신경망 모델 성능 비교
- 딥러닝 2개: 1D-CNN, Tabular Transformer 계열 모델
- ROC/AUC, actual-prediction, residual, QQ plot
- TableOne, t-SNE, UMAP/PCA, CCA, 상관분석
- SHAP summary/dependence/force plot 또는 대체 설명, LIME 형태 지역별 기여도
- 평가 조건 충족표 탭

## 참고

현재 앱은 설치 없이 바로 실행 가능한 패키지를 우선 사용합니다. `tableone`, `shap`, `umap-learn`이 설치되어 있지 않으면 앱 안에서 대체 시각화가 표시됩니다. 제출 전에는 `pip install -r requirements.txt`를 실행하면 실제 패키지 기반 화면으로 표시됩니다.