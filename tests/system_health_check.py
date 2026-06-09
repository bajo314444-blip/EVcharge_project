# ============================================================
# 파일명: system_health_check.py
# 설명: 도심/고속도로 EV 충전 프로젝트의 Two-Track 시스템
#       전체 헬스 체크(Health Check)를 수행하는 테스트 스크립트
#       (import(임포트), 데이터 로딩, ONNX 추론, 최적화, PDF 생성 검증)
# ============================================================

import sys  # sys(시스템) 모듈 import(임포트) — 경로 및 종료 코드 제어용
import os  # os(운영체제) 모듈 import(임포트) — 파일/디렉토리 경로 조작용

# --- 프로젝트 루트(root) 디렉토리 설정 및 sys.path 등록 블록 ---
# 실행 디렉토리에 관계없이 작업 디렉토리(CWD)를 프로젝트 루트로 강제 고정하고 sys.path 등록
script_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트(script)의 절대 디렉토리 경로를 취득
project_root = os.path.abspath(os.path.join(script_dir, ".."))  # 한 단계 상위 디렉토리를 project root(프로젝트 루트)로 설정
os.chdir(project_root)  # CWD(현재 작업 디렉토리)를 프로젝트 루트로 변경
if project_root not in sys.path:  # sys.path에 프로젝트 루트가 없으면 조건 진입
    sys.path.insert(0, project_root)  # sys.path 최상단에 프로젝트 루트를 삽입하여 모듈 검색 우선순위 확보

import json  # json(제이슨) 직렬화/역직렬화 모듈 import(임포트)
import pandas as pd  # pandas(판다스) 데이터 분석 라이브러리를 pd로 import(임포트)
import numpy as np  # numpy(넘파이) 수치 연산 라이브러리를 np로 import(임포트)

# --- 시스템 헬스 체크 메인 함수 ---
def run_health_check():  # 시스템 전체 상태를 6단계로 검증하는 함수 정의
    print("==================================================")  # 구분선 출력
    print("STARTING TWO-TRACK SYSTEM HEALTH CHECK (V4.3)...")  # 헬스 체크 시작 메시지 출력
    print("==================================================")  # 구분선 출력

    # --- [1/6] import(임포트) 테스트 블록 ---
    # 1. Test imports
    print("[1/6] Testing imports...")  # 1단계: 모듈 import(임포트) 테스트 시작 알림
    try:  # import(임포트) 예외 처리 시작
        from utils.data_processing import load_all_data, load_highway_data, DEFAULT_DATA_DIR, load_precomputed_analytics, ONNXModelWrapper  # 데이터 처리 유틸리티 모듈에서 주요 함수/상수 import(임포트)
        from utils.models import train_models, make_feature_matrix  # 모델 학습 유틸리티에서 함수 import(임포트)
        from utils.optimization import optimize_highway_chargers, calculate_single_region_trajectory  # 최적화 유틸리티에서 함수 import(임포트)
        from utils.visualizations import make_bubble_map, make_tableone, render_shap_or_fallback, render_highway_edge_plot  # 시각화 유틸리티에서 함수 import(임포트)
        from utils.pdf_generator import generate_report_pdf, generate_highway_report_pdf  # PDF 리포트 생성 유틸리티에서 함수 import(임포트)
        from views.urban_dashboard import render_dashboard, render_report  # 도심 대시보드(dashboard) 뷰에서 렌더링 함수 import(임포트)
        from views.highway_dashboard import render_highway_dashboard  # 고속도로 대시보드(dashboard) 뷰에서 렌더링 함수 import(임포트)
        from views.ai_assistant import render_ai_assistant  # AI 어시스턴트(assistant) 뷰에서 렌더링 함수 import(임포트)
        print(" -> All module imports: SUCCESS!")  # 모든 모듈 import(임포트) 성공 메시지 출력
    except Exception as e:  # import(임포트) 중 예외 발생 시 처리
        import traceback  # traceback(트레이스백) 모듈 import(임포트) — 상세 에러 출력용
        traceback.print_exc()  # 전체 스택 트레이스(stack trace)를 출력
        print(f" -> Import failed: {e}")  # 실패 원인 메시지 출력
        sys.exit(1)  # 종료 코드 1로 프로그램 즉시 종료

    # --- [2/6] 데이터 로딩 테스트 블록 ---
    # 2. Test data loading
    print("[2/6] Testing data loading from dataset/...")  # 2단계: 데이터 로딩 테스트 시작 알림
    try:  # 데이터 로딩 예외 처리 시작
        final_data, monthly_data, hourly_data = load_all_data(DEFAULT_DATA_DIR)  # 도심 전체 데이터(최종/월별/시간별)를 로드(load)
        print(f" -> Urban data: {len(final_data)} rows loaded.")  # 도심 최종 데이터 행(row) 수 출력
        print(f" -> Monthly data: {len(monthly_data)} rows loaded.")  # 월별 데이터 행(row) 수 출력

        hw_data = load_highway_data(DEFAULT_DATA_DIR)  # 고속도로 데이터를 로드(load)
        print(f" -> Highway data: {len(hw_data)} rows loaded.")  # 고속도로 데이터 행(row) 수 출력
        print(" -> Data loading: SUCCESS!")  # 데이터 로딩 성공 메시지 출력
    except Exception as e:  # 데이터 로딩 중 예외 발생 시 처리
        print(f" -> Data loading failed: {e}")  # 실패 원인 메시지 출력
        sys.exit(1)  # 종료 코드 1로 프로그램 즉시 종료

    # --- [3/6] Two-Track(ONNX + JSON) 로딩 테스트 블록 ---
    # 3. Test ONNX and JSON two-track loading
    print("[3/6] Testing Two-track (ONNX + JSON) loader...")  # 3단계: Two-Track 로더 테스트 시작 알림
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 루트 절대 경로를 재산출
    json_path = os.path.join(base_dir, "results", "precomputed_analytics.json")  # 사전 계산된 분석 결과 JSON 파일 경로 설정
    onnx_path = os.path.join(base_dir, "results", "best_model.onnx")  # 최적 모델 ONNX 파일 경로 설정

    if not os.path.exists(json_path) or not os.path.exists(onnx_path):  # JSON 또는 ONNX 파일이 존재하지 않으면 조건 진입
        print(f" -> Required model files not found at {json_path} or {onnx_path}! Please run save_models.py first.")  # 필수 파일 누락 경고 메시지 출력
        sys.exit(1)  # 종료 코드 1로 프로그램 즉시 종료

    try:  # Two-Track 로딩 예외 처리 시작
        model_state, model_state_smote = load_precomputed_analytics(json_path, onnx_path)  # JSON + ONNX 파일로부터 모델 상태를 로드(load)
        print(" -> Loaded precomputed analytics and ONNX model successfully.")  # 로드 성공 메시지 출력
        print(f" -> Best model: {model_state['best_name']}")  # 최적 모델 이름 출력
        print(f" -> Feature columns: {model_state['feature_columns']}")  # feature(피처) 컬럼(column) 목록 출력
        print(f" -> Metrics shape: {model_state['metrics'].shape}")  # 평가 지표 DataFrame(데이터프레임)의 shape(형태) 출력
        print(f" -> Importance shape: {model_state['importance'].shape}")  # 피처 중요도 DataFrame(데이터프레임)의 shape(형태) 출력
        if model_state_smote:  # SMOTE 적용 모델 상태가 존재하면 조건 진입
            print(f" -> SMOTE metrics shape: {model_state_smote['metrics'].shape}")  # SMOTE 모델 평가 지표 shape(형태) 출력
        print(" -> Two-track loading: SUCCESS!")  # Two-Track 로딩 성공 메시지 출력
    except Exception as e:  # Two-Track 로딩 중 예외 발생 시 처리
        import traceback  # traceback(트레이스백) 모듈 import(임포트)
        traceback.print_exc()  # 전체 스택 트레이스(stack trace) 출력
        print(f" -> Two-track loading failed: {e}")  # 실패 원인 메시지 출력
        sys.exit(1)  # 종료 코드 1로 프로그램 즉시 종료

    # --- [4/6] ONNX Wrapper(래퍼)를 이용한 예측 로직 테스트 블록 ---
    # 4. Test predictions with ONNX Wrapper
    print("[4/6] Testing prediction logic using ONNX runtime wrapper...")  # 4단계: ONNX 추론 테스트 시작 알림
    try:  # 예측 로직 예외 처리 시작
        best_name = model_state["best_name"]  # 최적 모델 이름을 추출
        best_model = model_state["models"][best_name]  # 최적 모델 객체(ONNXModelWrapper)를 가져옴
        X_test = model_state["X_test"]  # 테스트용 feature(피처) DataFrame(데이터프레임)을 가져옴

        # 배치(batch) 예측 shape(형태) 변환 테스트
        # Test shape conversion
        preds = best_model.predict(X_test)  # 전체 테스트 세트에 대해 배치(batch) predict(예측) 수행
        print(f" -> Batch predictions shape: {preds.shape} (Test set size: {len(X_test)})")  # 배치 예측 결과 shape(형태)와 테스트 세트 크기 출력
        print(f" -> Batch predictions sample: {preds[:5]}")  # 배치 예측 결과 처음 5개 샘플 출력

        # 단일 행(single-row) 예측 변환 테스트
        # Test single-row conversion
        single_row = X_test.iloc[0]  # 테스트 세트의 첫 번째 행을 Series(시리즈)로 추출
        single_pred = best_model.predict(single_row)  # 단일 행에 대해 predict(예측) 수행
        print(f" -> Single prediction output: {single_pred}")  # 단일 예측 결과 출력

        print(" -> ONNX Inference: SUCCESS!")  # ONNX 추론 성공 메시지 출력
    except Exception as e:  # ONNX 예측 중 예외 발생 시 처리
        import traceback  # traceback(트레이스백) 모듈 import(임포트)
        traceback.print_exc()  # 전체 스택 트레이스(stack trace) 출력
        print(f" -> ONNX Prediction failed: {e}")  # 실패 원인 메시지 출력
        sys.exit(1)  # 종료 코드 1로 프로그램 즉시 종료

    # --- [5/6] 고속도로 충전소 위치 최적화 알고리즘 테스트 블록 ---
    # 5. Test optimization algorithms
    print("[5/6] Testing highway location optimization...")  # 5단계: 최적화 테스트 시작 알림
    try:  # 최적화 예외 처리 시작
        if not hw_data.empty:  # 고속도로 데이터가 비어있지 않으면 조건 진입
            hw_sim = hw_data.copy()  # 원본 보호를 위해 고속도로 데이터를 deep copy(깊은 복사)
            hw_sim["부하_예측점수"] = 50.0  # 시뮬레이션용 부하 예측 점수를 50.0으로 일괄 설정
            hw_optimized = optimize_highway_chargers(hw_sim, budget=10)  # 예산 10 조건으로 고속도로 충전기 최적 배치 수행
            print(f" -> Highway optimization succeeded. Added chargers: {hw_optimized['최적_추가대수'].sum()}")  # 최적화 성공 및 추가 충전기 총 대수 출력
        print(" -> Optimization: SUCCESS!")  # 최적화 성공 메시지 출력
    except Exception as e:  # 최적화 중 예외 발생 시 처리
        print(f" -> Optimization failed: {e}")  # 실패 원인 메시지 출력
        sys.exit(1)  # 종료 코드 1로 프로그램 즉시 종료

    # --- [6/6] FPDF 기반 PDF 리포트 생성 테스트 블록 ---
    # 6. Test FPDF PDF report generation
    print("[6/6] Testing PDF generation logic...")  # 6단계: PDF 생성 테스트 시작 알림
    try:  # PDF 생성 예외 처리 시작
        import tempfile  # tempfile(임시 파일) 모듈 import(임포트)
        tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)  # .png 확장자의 임시 파일을 생성 (자동 삭제 비활성화)
        tmp_img_name = tmp_img.name  # 임시 파일의 절대 경로를 저장
        tmp_img.close()  # 임시 파일 핸들을 닫음

        # 테스트용 더미(dummy) 흰색 이미지를 저장
        # Save a dummy white image for testing
        from PIL import Image  # PIL(Pillow) 라이브러리에서 Image(이미지) 클래스를 import(임포트)
        img = Image.new('RGB', (100, 100), color = 'white')  # 100x100 크기의 흰색 RGB 이미지 객체 생성
        img.save(tmp_img_name)  # 생성한 이미지를 임시 파일 경로에 저장

        # 도심 리포트 PDF 생성 테스트
        # Generate urban report
        pdf_bytes = generate_report_pdf(  # 도심 분석 결과 PDF 리포트를 생성
            best_name=model_state["best_name"],  # 최적 모델 이름 전달
            test_rmse=0.5,  # 테스트 RMSE 값 전달
            top3_list=["1. Seoul Gangnam - Load 0.95", "2. Gyeonggi Anyang - Load 0.85"],  # 상위 3개 지역 리스트(list) 전달
            top_features=["total_ev_count", "infra_size_pca"],  # 주요 feature(피처) 이름 리스트(list) 전달
            feature_importance_img=tmp_img_name  # feature(피처) 중요도 이미지 경로 전달
        )  # generate_report_pdf 함수 호출 종료
        print(f" -> Urban PDF report generated: {len(pdf_bytes)} bytes.")  # 생성된 PDF 크기(bytes) 출력

        # 임시 파일 정리(cleanup)
        # Clean up
        os.remove(tmp_img_name)  # 테스트용 임시 이미지 파일 삭제
        print(" -> PDF report generation: SUCCESS!")  # PDF 생성 성공 메시지 출력
    except Exception as e:  # PDF 생성 중 예외 발생 시 처리
        import traceback  # traceback(트레이스백) 모듈 import(임포트)
        traceback.print_exc()  # 전체 스택 트레이스(stack trace) 출력
        print(f" -> PDF generation failed: {e}")  # 실패 원인 메시지 출력
        sys.exit(1)  # 종료 코드 1로 프로그램 즉시 종료

    print("==================================================")  # 구분선 출력
    print("SYSTEM HEALTH CHECK: ALL PASSED!")  # 모든 헬스 체크 통과 메시지 출력
    print("==================================================")  # 구분선 출력

if __name__ == "__main__":  # 이 스크립트(script)가 직접 실행될 때만 조건 진입
    run_health_check()  # 시스템 헬스 체크 메인 함수 실행
