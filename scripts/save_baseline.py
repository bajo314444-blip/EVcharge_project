# ============================================================
# 파일명: save_baseline.py
# 설명: Baseline(기준선) 모델을 학습하고 결과(메트릭, 특성 중요도, 예측값)를
#       CSV 파일로 저장하는 스크립트
# ============================================================

import os  # os 모듈 import(임포트) — 파일/디렉토리 경로 및 작업 디렉토리 조작용
import sys  # sys 모듈 import(임포트) — Python 경로(sys.path) 조작용

# --- 프로젝트 루트 경로 설정 블록 ---
# 실행 디렉토리에 관계없이 작업 디렉토리(CWD)를 프로젝트 루트로 강제 고정하고 sys.path 등록
script_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트 파일의 디렉토리 절대경로를 구함
project_root = os.path.abspath(os.path.join(script_dir, ".."))  # 스크립트 디렉토리의 상위(부모) 디렉토리를 프로젝트 루트(root)로 설정
os.chdir(project_root)  # 작업 디렉토리(CWD)를 프로젝트 루트로 변경
if project_root not in sys.path:  # 프로젝트 루트가 sys.path에 없으면
    sys.path.insert(0, project_root)  # sys.path 맨 앞에 프로젝트 루트를 추가하여 모듈 import(임포트) 가능하게 설정

# --- 외부 라이브러리 및 유틸리티 모듈 import(임포트) ---
import pandas as pd  # pandas(판다스) 라이브러리를 pd로 import(임포트) — 데이터 처리용
from utils.data_processing import load_all_data, DEFAULT_DATA_DIR  # 데이터 로딩 함수와 기본 데이터 디렉토리 상수를 import(임포트)
from utils.models import train_models  # 모델 학습 함수 train_models를 import(임포트)

# --- Baseline(기준선) 모델 학습 및 결과 저장 함수 ---
def save_baseline_results():  # baseline(기준선) 결과를 저장하는 메인 함수 정의
    print("Loading data...")  # 데이터 로딩 시작 메시지 출력
    final_data, _, _ = load_all_data(DEFAULT_DATA_DIR)  # 기본 데이터 디렉토리에서 전체 데이터를 로드하고 final_data에 저장 (나머지 반환값은 무시)
    
    print("Training baseline models...")  # baseline(기준선) 모델 학습 시작 메시지 출력
    model_state = train_models(final_data.to_json(orient="split"))  # DataFrame(데이터프레임)을 JSON 문자열로 변환 후 모델 학습, 결과를 model_state에 저장
    
    # --- 결과 저장 디렉토리 생성 블록 ---
    os.makedirs("results/baseline", exist_ok=True)  # results/baseline 디렉토리를 생성 (이미 존재하면 무시)
    
    # --- 모델 성능 메트릭(metrics) 저장 블록 ---
    metrics = model_state["metrics"]  # model_state dictionary(딕셔너리)에서 성능 메트릭 DataFrame(데이터프레임)을 추출
    metrics.to_csv("results/baseline/baseline_metrics.csv", index=False)  # 메트릭을 CSV 파일로 저장 (index(인덱스) 제외)
    
    # --- Feature Importance(특성 중요도) 저장 블록 ---
    importance = model_state["importance"]  # model_state에서 Feature Importance(특성 중요도) DataFrame(데이터프레임)을 추출
    importance.to_csv("results/baseline/baseline_feature_importance.csv", index=False)  # 특성 중요도를 CSV 파일로 저장 (index(인덱스) 제외)
    
    # --- 테스트 세트 예측값 저장 블록 (ROC/AUC 계산용) ---
    predictions = model_state["predictions"]  # model_state에서 예측 결과 DataFrame(데이터프레임)을 추출
    predictions.to_csv("results/baseline/baseline_predictions.csv", index=False)  # 예측값을 CSV 파일로 저장 (index(인덱스) 제외)
    
    print("Baseline results saved successfully in 'results/baseline/' directory.")  # 결과 저장 완료 메시지 출력

# --- 스크립트 직접 실행 시 진입점 ---
if __name__ == "__main__":  # 이 파일이 직접 실행될 때만 아래 코드 실행
    save_baseline_results()  # baseline(기준선) 결과 저장 함수 호출
