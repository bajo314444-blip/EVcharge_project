# ============================================================
# 파일명: save_models.py
# 설명: 전체 파이프라인을 실행하여 모델 학습, 임베딩 사전계산,
#       학술 강건성 평가(Bootstrap, Nested CV, Adversarial, Ablation,
#       DCA, Spatial CV, Survival), ONNX 내보내기, JSON 직렬화를 수행하는 스크립트
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

# --- 외부 라이브러리 및 유틸리티 모듈 import(임포트) 블록 ---
import json  # json 모듈 import(임포트) — JSON 직렬화/역직렬화용
import pandas as pd  # pandas(판다스) 라이브러리를 pd로 import(임포트) — 데이터 처리용
import numpy as np  # numpy(넘파이) 라이브러리를 np로 import(임포트) — 수치 연산용
from utils.data_processing import load_all_data, DEFAULT_DATA_DIR  # 데이터 로딩 함수와 기본 데이터 디렉토리 상수를 import(임포트)
from utils.models import train_models  # 모델 학습 함수 train_models를 import(임포트)
from sklearn.preprocessing import StandardScaler  # StandardScaler(표준화 스케일러) 클래스를 import(임포트)
from sklearn.manifold import TSNE  # t-SNE(t-분산 확률적 이웃 임베딩) 클래스를 import(임포트)
from sklearn.decomposition import PCA  # PCA(주성분분석) 클래스를 import(임포트)
from sklearn.cross_decomposition import CCA  # CCA(정준상관분석) 클래스를 import(임포트)
from sklearn.metrics import roc_curve, auc  # ROC Curve(ROC 곡선) 및 AUC(곡선 아래 면적) 함수를 import(임포트)
from utils.optimization import (  # 최적화/평가 유틸리티 모듈에서 다수의 함수를 import(임포트)
    calculate_bootstrap_ci,  # Bootstrap(부트스트랩) 신뢰구간 계산 함수
    run_adversarial_attack,  # Adversarial Attack(적대적 공격) 실행 함수
    run_ablation_study,  # Ablation Study(제거 연구) 실행 함수
    calculate_dca,  # DCA(Decision Curve Analysis, 의사결정 곡선 분석) 계산 함수
    run_nested_cv,  # Nested CV(중첩 교차검증) 실행 함수
    run_spatial_external_validation,  # Spatial External Validation(공간적 외부 검증) 실행 함수
    run_survival_simulation  # Survival Simulation(생존 분석 시뮬레이션) 실행 함수
)

# --- JSON 직렬화 함수 ---
def serialize_to_json(model_state, model_state_smote, json_path):  # 모델 상태를 JSON 파일로 직렬화하는 함수 정의
    # --- numpy/pandas 타입을 JSON 직렬화 가능한 형식으로 변환하는 헬퍼(helper) 함수 ---
    def default_converter(obj):  # JSON 직렬화 시 기본 변환기 함수 정의
        if isinstance(obj, np.integer):  # numpy integer(정수) 타입인 경우
            return int(obj)  # Python int(정수)로 변환하여 반환
        elif isinstance(obj, np.floating):  # numpy floating(부동소수점) 타입인 경우
            return float(obj)  # Python float(실수)로 변환하여 반환
        elif isinstance(obj, np.ndarray):  # numpy ndarray(배열) 타입인 경우
            return obj.tolist()  # Python list(리스트)로 변환하여 반환
        elif isinstance(obj, pd.DataFrame):  # pandas DataFrame(데이터프레임) 타입인 경우
            return obj.to_dict(orient="records")  # records(레코드) 형식 dictionary(딕셔너리) list(리스트)로 변환하여 반환
        elif isinstance(obj, pd.Series):  # pandas Series(시리즈) 타입인 경우
            return obj.tolist()  # Python list(리스트)로 변환하여 반환
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")  # 지원하지 않는 타입이면 TypeError(타입 에러) 발생

    # --- 기본 모델 데이터를 dictionary(딕셔너리)로 구성 ---
    data = {  # JSON으로 저장할 데이터 dictionary(딕셔너리) 생성
        "metrics": model_state["metrics"].to_dict(orient="records"),  # 성능 메트릭을 records(레코드) 형식으로 변환
        "importance": model_state["importance"].to_dict(orient="records"),  # Feature Importance(특성 중요도)를 records(레코드) 형식으로 변환
        "predictions": model_state["predictions"].to_dict(orient="records"),  # 예측 결과를 records(레코드) 형식으로 변환
        "best_name": model_state["best_name"],  # 최적 모델명 저장
        "feature_columns": model_state["feature_columns"],  # Feature(특성) 컬럼 목록 저장
        "model_groups": model_state["model_groups"],  # 모델 그룹 매핑 정보 저장
        "X": model_state["X"].to_dict(orient="records"),  # 전체 Feature Matrix(특성 행렬)를 records(레코드) 형식으로 변환
        "y": model_state["y"].tolist(),  # 전체 타겟 변수를 list(리스트)로 변환
        "X_train": model_state["X_train"].to_dict(orient="records"),  # 학습 Feature Matrix(특성 행렬)를 records(레코드) 형식으로 변환
        "y_train": model_state["y_train"].tolist(),  # 학습 타겟 변수를 list(리스트)로 변환
        "X_test": model_state["X_test"].to_dict(orient="records"),  # 테스트 Feature Matrix(특성 행렬)를 records(레코드) 형식으로 변환
        "y_test": model_state["y_test"].tolist(),  # 테스트 타겟 변수를 list(리스트)로 변환
    }
    
    # --- Embedding(임베딩) 사전계산 결과 저장 블록 ---
    if "precomputed_tsne_xy" in model_state:  # t-SNE 사전계산 결과가 있으면
        data["precomputed_tsne_xy"] = model_state["precomputed_tsne_xy"].tolist()  # t-SNE 좌표를 list(리스트)로 변환하여 저장
    if "precomputed_umap_xy" in model_state:  # UMAP 사전계산 결과가 있으면
        data["precomputed_umap_xy"] = model_state["precomputed_umap_xy"].tolist()  # UMAP 좌표를 list(리스트)로 변환하여 저장
        data["precomputed_umap_title"] = model_state.get("precomputed_umap_title", "UMAP")  # UMAP 제목 저장 (기본값 "UMAP")
    if "precomputed_cca_x_c" in model_state and "precomputed_cca_y_c" in model_state:  # CCA 사전계산 결과가 있으면
        data["precomputed_cca_x_c"] = model_state["precomputed_cca_x_c"].tolist()  # CCA X측 정준 점수를 list(리스트)로 변환하여 저장
        data["precomputed_cca_y_c"] = model_state["precomputed_cca_y_c"].tolist()  # CCA Y측 정준 점수를 list(리스트)로 변환하여 저장
        
    # --- Bootstrap(부트스트랩) 신뢰구간 결과 저장 블록 ---
    if "precomputed_bootstrap" in model_state:  # Bootstrap(부트스트랩) 사전계산 결과가 있으면
        pb = model_state["precomputed_bootstrap"]  # Bootstrap(부트스트랩) 결과 tuple(튜플)을 pb에 할당
        data["precomputed_bootstrap"] = {  # Bootstrap(부트스트랩) 결과를 dictionary(딕셔너리)로 구성
            "ci_rmse": [float(v) for v in pb[0]],  # RMSE 신뢰구간을 float(실수) list(리스트)로 변환
            "ci_r2": [float(v) for v in pb[1]],  # R² 신뢰구간을 float(실수) list(리스트)로 변환
            "bootstrap_rmse": [float(v) for v in pb[2]],  # Bootstrap(부트스트랩) RMSE 분포를 float(실수) list(리스트)로 변환
            "bootstrap_r2": [float(v) for v in pb[3]]  # Bootstrap(부트스트랩) R² 분포를 float(실수) list(리스트)로 변환
        }
        
    # --- Nested CV(중첩 교차검증) 결과 저장 블록 ---
    if "precomputed_nested_cv" in model_state:  # Nested CV(중첩 교차검증) 사전계산 결과가 있으면
        pnc = model_state["precomputed_nested_cv"]  # Nested CV 결과 tuple(튜플)을 pnc에 할당
        data["precomputed_nested_cv"] = {  # Nested CV 결과를 dictionary(딕셔너리)로 구성
            "mean_rmse": float(pnc[0]),  # 평균 RMSE를 float(실수)로 변환
            "std_rmse": float(pnc[1]),  # RMSE 표준편차를 float(실수)로 변환
            "outer_scores": [float(v) for v in pnc[2]]  # 외부 폴드(fold) 점수들을 float(실수) list(리스트)로 변환
        }
        
    # --- Adversarial Attack(적대적 공격) 결과 저장 블록 ---
    if "precomputed_adversarial" in model_state:  # Adversarial(적대적 공격) 사전계산 결과가 있으면
        pa = model_state["precomputed_adversarial"]  # Adversarial 결과 DataFrame(데이터프레임)을 pa에 할당
        data["precomputed_adversarial"] = pa.to_dict(orient="records")  # records(레코드) 형식으로 변환하여 저장
        
    # --- Ablation Study(제거 연구) 결과 저장 블록 ---
    if "precomputed_ablation" in model_state:  # Ablation(제거 연구) 사전계산 결과가 있으면
        pabl = model_state["precomputed_ablation"]  # Ablation 결과 DataFrame(데이터프레임)을 pabl에 할당
        data["precomputed_ablation"] = pabl.to_dict(orient="records")  # records(레코드) 형식으로 변환하여 저장
        
    # --- DCA(의사결정 곡선 분석) 결과 저장 블록 ---
    if "precomputed_dca" in model_state:  # DCA 사전계산 결과가 있으면
        pdca = model_state["precomputed_dca"]  # DCA 결과 DataFrame(데이터프레임)을 pdca에 할당
        data["precomputed_dca"] = pdca.to_dict(orient="records")  # records(레코드) 형식으로 변환하여 저장
        
    # --- Spatial CV(공간적 외부 검증) 결과 저장 블록 ---
    if "precomputed_spatial" in model_state:  # Spatial(공간적) 검증 사전계산 결과가 있으면
        ps = model_state["precomputed_spatial"]  # Spatial 검증 결과 dictionary(딕셔너리)를 ps에 할당
        spatial_serializable = {}  # JSON 직렬화 가능한 빈 dictionary(딕셔너리) 초기화
        for region, metrics_tuple in ps.items():  # 각 지역과 해당 메트릭 tuple(튜플)에 대해 순회
            spatial_serializable[region] = [  # 각 지역의 메트릭을 list(리스트)로 구성
                float(metrics_tuple[0]) if metrics_tuple[0] is not None else None,  # RMSE 값을 float(실수)로 변환 (None이면 None 유지)
                float(metrics_tuple[1]) if metrics_tuple[1] is not None else None,  # MAE 값을 float(실수)로 변환 (None이면 None 유지)
                float(metrics_tuple[2]) if metrics_tuple[2] is not None else None,  # R² 값을 float(실수)로 변환 (None이면 None 유지)
                None  # 에러 메시지 자리(placeholder) — 항상 None으로 설정
            ]
        data["precomputed_spatial"] = spatial_serializable  # Spatial(공간적) 검증 결과를 데이터에 저장
        
    # --- Survival Simulation(생존 분석 시뮬레이션) 결과 저장 블록 ---
    if "precomputed_survival_5" in model_state:  # Survival(생존 분석) 사전계산 결과가 있으면
        psurv = model_state["precomputed_survival_5"]  # Survival 결과 DataFrame(데이터프레임)을 psurv에 할당
        data["precomputed_survival_5"] = psurv.to_dict(orient="records")  # records(레코드) 형식으로 변환하여 저장

    # --- ROC Curve(ROC 곡선) 데이터 저장 블록 ---
    if "precomputed_roc_data" in model_state:  # ROC 사전계산 데이터가 있으면
        data["precomputed_roc_data"] = model_state["precomputed_roc_data"]  # ROC 데이터를 그대로 저장
        
    # --- SMOTE 모델 메트릭 저장 블록 ---
    if model_state_smote is not None:  # SMOTE 적용 모델 상태가 존재하면
        data["model_state_smote_metrics"] = model_state_smote["metrics"].to_dict(orient="records")  # SMOTE 메트릭을 records(레코드) 형식으로 변환하여 저장
        data["model_state_smote_best_name"] = model_state_smote["best_name"]  # SMOTE 최적 모델명 저장
        
    # --- JSON 파일 쓰기 블록 ---
    with open(json_path, 'w', encoding='utf-8') as f:  # JSON 파일을 UTF-8 encoding(인코딩)으로 쓰기 모드 열기
        json.dump(data, f, ensure_ascii=False, indent=4, default=default_converter)  # 데이터를 JSON으로 직렬화하여 파일에 저장 (한글 유지, 4칸 들여쓰기)

# --- 학습된 모델 내보내기 메인 함수 ---
def export_trained_models():  # 전체 파이프라인을 실행하는 메인 함수 정의
    # --- 1단계: 원본 데이터 로딩 ---
    print("[1/5] Loading raw data...")  # 원본 데이터 로딩 시작 메시지 출력
    final_data, _, _ = load_all_data(DEFAULT_DATA_DIR)  # 기본 데이터 디렉토리에서 전체 데이터를 로드
    final_json = final_data.to_json(orient="split")  # DataFrame(데이터프레임)을 split 형식 JSON 문자열로 변환
    
    # --- 2단계: Baseline(기준선) 모델 학습 (Pre-SMOTE) ---
    print("[2/5] Training baseline models (Pre-SMOTE 7 models)...")  # Pre-SMOTE baseline 모델 학습 시작 메시지 출력
    model_state = train_models(final_json, use_smote=False)  # SMOTE 미적용으로 7개 모델 학습, 결과를 model_state에 저장
    
    # --- 2.3단계: Embedding(임베딩) 사전계산 블록 (t-SNE, UMAP, CCA) ---
    print("[2.3/5] Pre-computing embeddings (t-SNE, UMAP, CCA)...")  # Embedding(임베딩) 사전계산 시작 메시지 출력
    try:  # 예외 발생 가능 구간 시작
        X_embed = model_state["X"].copy()  # 전체 Feature Matrix(특성 행렬)의 복사본 생성
        scaled_embed = StandardScaler().fit_transform(X_embed)  # StandardScaler(표준화 스케일러)로 특성을 표준화
        
        # --- t-SNE(t-분산 확률적 이웃 임베딩) 사전계산 ---
        perplexity = max(5, min(20, len(X_embed) // 4))  # t-SNE perplexity(혼잡도)를 데이터 크기에 맞게 동적 설정 (5~20 범위)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", learning_rate="auto")  # t-SNE 객체 생성 (2차원, PCA 초기화, 자동 학습률)
        tsne_xy = tsne.fit_transform(scaled_embed)  # t-SNE 변환 수행하여 2차원 좌표 생성
        model_state["precomputed_tsne_xy"] = tsne_xy  # t-SNE 결과를 model_state에 저장
        
        # --- UMAP 또는 PCA fallback(대체) 사전계산 ---
        try:  # UMAP import(임포트) 시도
            import umap  # umap 라이브러리 import(임포트) 시도
            reducer = umap.UMAP(random_state=42)  # UMAP reducer(차원 축소기) 객체 생성 (random_state 고정)
            umap_xy = reducer.fit_transform(scaled_embed)  # UMAP 변환 수행하여 2차원 좌표 생성
            model_state["precomputed_umap_title"] = "UMAP"  # 제목을 "UMAP"으로 설정
        except Exception:  # UMAP import(임포트) 실패 시
            reducer = PCA(n_components=2, random_state=42)  # PCA(주성분분석)로 fallback(대체) — 2차원 축소
            umap_xy = reducer.fit_transform(scaled_embed)  # PCA 변환 수행하여 2차원 좌표 생성
            model_state["precomputed_umap_title"] = "UMAP 패키지 미설치 시 PCA 대체 시각화"  # 제목에 PCA 대체 사용 안내 설정
        model_state["precomputed_umap_xy"] = umap_xy  # UMAP/PCA 결과를 model_state에 저장
        
        # --- CCA(정준상관분석) 사전계산 ---
        cca_x = StandardScaler().fit_transform(final_data[["전기차_전체대수", "충전인프라_규모_PCA", "충전기_1대당_평균용량", "인프라_부하지수"]])  # CCA X측 변수 4개를 표준화
        cca_y = StandardScaler().fit_transform(final_data[["전력_부하지수"]])  # CCA Y측 타겟 변수(전력_부하지수)를 표준화
        cca = CCA(n_components=1)  # CCA(정준상관분석) 객체 생성 (1개 정준 성분)
        x_c, y_c = cca.fit_transform(cca_x, cca_y)  # CCA 학습 및 변환 수행
        model_state["precomputed_cca_x_c"] = x_c[:, 0]  # X측 정준 점수(canonical score)를 model_state에 저장
        model_state["precomputed_cca_y_c"] = y_c[:, 0]  # Y측 정준 점수(canonical score)를 model_state에 저장
        
    except Exception as e:  # Embedding(임베딩) 사전계산 중 예외 발생 시
        print(f"Embedding pre-computation failed: {e}")  # 에러 메시지 출력
        
    # --- 2.7단계: 학술 강건성 평가 사전계산 블록 ---
    print("[2.7/5] Pre-computing robust evaluations (Nested CV, Bootstrap, DCA, etc.) to eliminate lag...")  # 강건성 평가 사전계산 시작 메시지 출력
    try:  # 예외 발생 가능 구간 시작
        best_name = model_state["best_name"]  # 최적 모델명 추출
        best_model = model_state["models"][best_name]  # 최적 모델 객체 추출
        X_train, X_test = model_state["X_train"], model_state["X_test"]  # 학습/테스트 Feature Matrix(특성 행렬) 참조
        y_train, y_test = model_state["y_train"], model_state["y_test"]  # 학습/테스트 타겟 변수 참조
        X, y = model_state["X"], model_state["y"]  # 전체 Feature Matrix(특성 행렬)와 타겟 변수 참조
        importance = model_state["importance"]  # Feature Importance(특성 중요도) 참조
        
        # --- Bootstrap(부트스트랩) 신뢰구간 계산 ---
        print(" -> Computing Bootstrap CI...")  # Bootstrap(부트스트랩) CI(신뢰구간) 계산 시작 메시지 출력
        model_state["precomputed_bootstrap"] = calculate_bootstrap_ci(best_model, X_test, y_test, n_iterations=100)  # 100회 반복 Bootstrap(부트스트랩) 신뢰구간 계산 후 저장
        
        # --- Nested CV(중첩 교차검증) 계산 ---
        print(" -> Computing Nested 10-fold CV (this may take a few seconds)...")  # Nested 10-fold CV 계산 시작 메시지 출력
        model_state["precomputed_nested_cv"] = run_nested_cv(best_model, X, y)  # Nested CV(중첩 교차검증) 실행 후 결과 저장
        
        # --- Adversarial Attack(적대적 공격) 계산 ---
        print(" -> Computing Adversarial Attack...")  # Adversarial Attack(적대적 공격) 계산 시작 메시지 출력
        model_state["precomputed_adversarial"] = run_adversarial_attack(best_model, X_test, y_test)  # Adversarial Attack(적대적 공격) 실행 후 결과 저장
        
        # --- Ablation Study(제거 연구) 계산 ---
        print(" -> Computing Ablation Study...")  # Ablation Study(제거 연구) 계산 시작 메시지 출력
        model_state["precomputed_ablation"] = run_ablation_study(best_model, X_train, y_train, X_test, y_test, importance)  # Ablation Study(제거 연구) 실행 후 결과 저장
        
        # --- DCA(의사결정 곡선 분석) 계산 ---
        print(" -> Computing DCA...")  # DCA(의사결정 곡선 분석) 계산 시작 메시지 출력
        model_state["precomputed_dca"] = calculate_dca(best_model, X_test, y_test)  # DCA 계산 후 결과 저장
        
        # --- Spatial External Validation(공간적 외부 검증) 계산 ---
        print(" -> Computing Spatial External Validations...")  # Spatial(공간적) 외부 검증 계산 시작 메시지 출력
        spatial_results = {}  # 공간적 검증 결과를 저장할 빈 dictionary(딕셔너리) 초기화
        for region in ["인천", "서울", "경기"]:  # 인천, 서울, 경기 3개 지역에 대해 순회
            ext_rmse, ext_mae, ext_r2, ext_err = run_spatial_external_validation(best_model, X, y, region)  # 해당 지역 holdout(홀드아웃) 방식 외부 검증 실행
            spatial_results[region] = (ext_rmse, ext_mae, ext_r2, ext_err)  # 결과 tuple(튜플)을 지역별로 저장
        model_state["precomputed_spatial"] = spatial_results  # Spatial(공간적) 검증 결과를 model_state에 저장
        
        # --- Survival Analysis(생존 분석) 시뮬레이션 계산 (성장률 5%) ---
        print(" -> Computing Survival Curve...")  # Survival Curve(생존 곡선) 계산 시작 메시지 출력
        surv_res = run_survival_simulation(final_data, growth_rate=0.05)  # 연간 5% 성장률로 Survival(생존) 시뮬레이션 실행
        model_state["precomputed_survival_5"] = surv_res  # Survival(생존 분석) 결과를 model_state에 저장
        
        # --- ROC Curve(ROC 곡선) 사전계산 블록 ---
        print(" -> Precomputing ROC curves...")  # ROC Curve(ROC 곡선) 사전계산 시작 메시지 출력
        threshold = y.quantile(0.7)  # 타겟 변수의 70% quantile(분위수)을 threshold(임계값)으로 설정
        y_test_binary = (y_test >= threshold).astype(int)  # 테스트 타겟값을 threshold(임계값) 기준으로 이진(binary) 분류로 변환
        roc_data = {}  # ROC 데이터를 저장할 빈 dictionary(딕셔너리) 초기화
        for name, model in model_state["models"].items():  # 모든 모델에 대해 순회
            score = model.predict(X_test)  # 테스트 데이터에 대한 예측 점수 계산
            fpr, tpr, _ = roc_curve(y_test_binary, score)  # FPR(위양성률), TPR(진양성률) 계산
            roc_data[name] = {  # 각 모델의 ROC 데이터를 dictionary(딕셔너리)로 구성
                "fpr": fpr.tolist(),  # FPR(위양성률) 배열을 list(리스트)로 변환
                "tpr": tpr.tolist(),  # TPR(진양성률) 배열을 list(리스트)로 변환
                "auc": float(auc(fpr, tpr)),  # AUC(곡선 아래 면적)를 float(실수)로 변환
                "group": model_state["model_groups"][name]  # 해당 모델의 그룹 정보 저장
            }
        model_state["precomputed_roc_data"] = roc_data  # ROC 데이터를 model_state에 저장
        
    except Exception as e:  # 강건성 사전계산 중 예외 발생 시
        print(f"Robustness pre-computation failed: {e}")  # 에러 메시지 출력
        
    # --- 3단계: SMOTE 적용 모델 학습 (Post-SMOTE) ---
    print("[3/5] Training balanced models (Post-SMOTE 7 models)...")  # Post-SMOTE 균형 모델 학습 시작 메시지 출력
    model_state_smote = train_models(final_json, use_smote=True)  # SMOTE 적용하여 7개 모델 학습
    
    os.makedirs("results", exist_ok=True)  # results 디렉토리 생성 (이미 존재하면 무시)
    
    # --- 3.5단계: ONNX Format(형식)으로 모델 내보내기 ---
    print("[3.5/5] Exporting best model to ONNX...")  # ONNX 모델 내보내기 시작 메시지 출력
    try:  # 예외 발생 가능 구간 시작
        from skl2onnx import convert_sklearn  # skl2onnx에서 sklearn 모델 변환 함수를 import(임포트)
        from skl2onnx.common.data_types import FloatTensorType  # ONNX input(입력) 타입 정의 클래스를 import(임포트)
        
        best_name = model_state["best_name"]  # 최적 모델명 추출
        best_model = model_state["models"][best_name]  # 최적 모델 객체 추출
        
        initial_type = [('float_input', FloatTensorType([None, 13]))]  # ONNX input(입력) 타입 정의: 13개 Feature(특성), batch(배치) 크기 가변
        
        try:  # ONNX 변환 시도
            onnx_model = convert_sklearn(best_model, initial_types=initial_type, target_opset=12)  # 최적 모델을 ONNX 형식으로 변환 (opset 12)
        except Exception as conversion_error:  # 변환 실패 시
            print(f"ONNX conversion for best model {best_name} failed: {conversion_error}. Falling back to RandomForest (Tuned)...")  # 변환 실패 메시지 출력 및 RandomForest(Tuned)로 fallback(대체) 안내
            best_model = model_state["models"]["RandomForest (Tuned)"]  # RandomForest(Tuned) 모델로 fallback(대체)
            onnx_model = convert_sklearn(best_model, initial_types=initial_type, target_opset=12)  # fallback(대체) 모델을 ONNX로 변환
            model_state["best_name"] = "RandomForest (Tuned)"  # 최적 모델명을 fallback(대체) 모델로 업데이트
            
        onnx_path = "results/best_model.onnx"  # ONNX 모델 저장 경로 설정
        with open(onnx_path, "wb") as f:  # ONNX 파일을 binary(바이너리) 쓰기 모드로 열기
            f.write(onnx_model.SerializeToString())  # ONNX 모델을 직렬화하여 파일에 저장
        print(f"[SUCCESS] ONNX model successfully saved to '{onnx_path}'")  # ONNX 저장 성공 메시지 출력
    except Exception as e:  # ONNX 내보내기 실패 시
        print(f"[ERROR] Failed to export ONNX model: {e}")  # 에러 메시지 출력

    # --- 4단계: JSON Format(형식)으로 전체 데이터 내보내기 ---
    json_path = "results/precomputed_analytics.json"  # JSON 분석 결과 저장 경로 설정
    print(f"[4/5] Serializing all visualization and academic metrics to '{json_path}'...")  # JSON 직렬화 시작 메시지 출력
    try:  # 예외 발생 가능 구간 시작
        serialize_to_json(model_state, model_state_smote, json_path)  # 모델 상태와 SMOTE 메트릭을 JSON으로 직렬화
        print(f"[SUCCESS] Metrics successfully saved to '{json_path}'")  # JSON 저장 성공 메시지 출력
    except Exception as e:  # JSON 내보내기 실패 시
        print(f"[ERROR] Failed to export JSON analytics: {e}")  # 에러 메시지 출력

    print("[SUCCESS] Two-track models, embeddings, and robustness metrics exported successfully!")  # 전체 파이프라인 성공 완료 메시지 출력

# --- 스크립트 직접 실행 시 진입점 ---
if __name__ == "__main__":  # 이 파일이 직접 실행될 때만 아래 코드 실행
    export_trained_models()  # 전체 모델 내보내기 함수 호출
