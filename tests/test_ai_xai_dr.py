# ============================================================
# 파일명: test_ai_xai_dr.py
# 설명: AI 관제비서(AI Assistant) 핵심 기능 단위 테스트(Unit Test) 스크립트.
#       시뮬레이션, SHAP(설명가능 AI), DR(수요반응) 시뮬레이션,
#       상위 지역 조회 함수의 정상 동작을 검증한다.
# ============================================================

import streamlit as st  # streamlit(스트림릿) 웹 앱 프레임워크를 st로 import(임포트) — session_state(세션 상태) 모킹용
import os  # os(운영체제) 모듈을 import(임포트) — 경로 조작용
import sys  # sys(시스템) 모듈을 import(임포트) — Python 경로 및 stdout(표준출력) 제어용

# --- 프로젝트 루트(root) 경로 설정 블록 ---
# 실행 디렉토리에 관계없이 작업 디렉토리(CWD)를 프로젝트 루트로 강제 고정하고 sys.path 등록
script_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 테스트 스크립트 파일의 디렉터리 절대경로를 구함
project_root = os.path.abspath(os.path.join(script_dir, ".."))  # 스크립트 상위 디렉터리를 project root(프로젝트 루트)로 설정
os.chdir(project_root)  # CWD(현재 작업 디렉토리)를 프로젝트 루트로 변경
if project_root not in sys.path:  # sys.path에 프로젝트 루트가 없으면
    sys.path.insert(0, project_root)  # sys.path 최상단에 프로젝트 루트를 삽입하여 모듈 import(임포트) 가능하게 설정

from utils.data_processing import load_all_data, load_precomputed_analytics  # 데이터 로드 및 사전계산 모델 복원 함수 import(임포트)
from views.ai_assistant import get_simulation_result, get_shap_analysis, run_dr_simulation, get_top_regions  # AI 관제비서 핵심 함수들 import(임포트)

# --- AI/XAI/DR 기능 단위 테스트 메인 함수 ---
def main():  # 테스트 시나리오를 순차 실행하는 main(메인) 함수 정의
    print("="*60)  # 테스트 시작 구분선 출력
    print("Running AI Assistant Features Unit Test (SHAP & DR Sim)")  # 테스트 제목(영문) 출력
    print("="*60)  # 구분선 출력
    
    # --- 1. 데이터 로드 및 세션 상태(session_state) 설정 ---
    dataset_dir = "./dataset"  # 테스트용 dataset(데이터셋) 디렉터리 경로 지정
    print(f"Loading dataset from: {dataset_dir}")  # 데이터 로드 경로를 콘솔에 출력
    final_data, monthly_data, hourly_data = load_all_data(dataset_dir)  # 도심 통합 데이터(최종/월별/시간별) 로드
    st.session_state["final_data"] = final_data  # final_data를 session_state(세션 상태)에 저장하여 AI 함수가 접근 가능하게 설정
    
    print("Columns in final_data:")  # final_data 컬럼 목록 출력 안내
    columns_str = ", ".join(final_data.columns.tolist())  # 컬럼명 리스트를 쉼표 구분 문자열로 결합
    sys.stdout.buffer.write((columns_str + "\n").encode('utf-8'))  # UTF-8 인코딩으로 컬럼 목록을 stdout(표준출력)에 출력
    
    print("Unique regions containing '안양':")  # '안양' 포함 지역명 출력 안내
    anyang_regions = final_data[final_data["지역"].str.contains("안양", na=False)]["지역"].unique().tolist()  # '안양' 포함 지역의 고유 지역명 리스트 추출
    sys.stdout.buffer.write((", ".join(anyang_regions) + "\n").encode('utf-8'))  # 안양 관련 지역명을 UTF-8로 stdout(표준출력)에 출력
    
    json_path = os.path.join("results", "precomputed_analytics.json")  # 사전계산 분석 JSON 파일 경로 생성
    onnx_path = os.path.join("results", "best_model.onnx")  # 사전학습 ONNX 모델 파일 경로 생성
    
    print(f"Loading precomputed model states: {json_path}, {onnx_path}")  # 모델 파일 로드 경로 출력
    model_state, model_state_smote = load_precomputed_analytics(json_path, onnx_path)  # JSON+ONNX 이원화 아키텍처로 모델 상태 복원
    st.session_state["model_state"] = model_state  # model_state를 session_state(세션 상태)에 저장
    
    # --- 2. get_simulation_result(시뮬레이션 결과 조회) 테스트 ---
    print("\n--- 1. Testing get_simulation_result ---")  # 시뮬레이션 테스트 섹션 헤더 출력
    sim_result = get_simulation_result("안양시", 10, "급속")  # 안양시에 급속 충전기 10대 증설 시뮬레이션 실행
    sys.stdout.buffer.write(sim_result.encode('utf-8'))  # 시뮬레이션 결과 텍스트를 UTF-8로 stdout(표준출력)에 출력
    print("\n" + "-"*40)  # 섹션 구분선 출력
    
    # --- 3. get_shap_analysis(SHAP 분석) 테스트 ---
    print("\n--- 2. Testing get_shap_analysis ---")  # SHAP 분석 테스트 섹션 헤더 출력
    shap_result = get_shap_analysis("안양시")  # 안양시 지역에 대한 SHAP(설명가능 AI) 분석 실행
    sys.stdout.buffer.write(shap_result.encode('utf-8'))  # SHAP 분석 결과 텍스트를 UTF-8로 stdout(표준출력)에 출력
    print("\n" + "-"*40)  # 섹션 구분선 출력
    
    # --- 세션 상태(session_state)에 SHAP 차트 데이터 저장 여부 검증 ---
    shap_chart = st.session_state.get("last_shap_chart")  # session_state(세션 상태)에서 마지막 SHAP 차트 데이터 조회
    print(f"last_shap_chart stored in st.session_state: {shap_chart is not None}")  # SHAP 차트 저장 여부를 콘솔에 출력
    if shap_chart:  # SHAP 차트 데이터가 존재하는 경우
        print(f"Keys: {list(shap_chart.keys())}")  # 차트 dict(딕셔너리)의 key(키) 목록 출력
        print(f"Region: {shap_chart.get('region')}")  # 분석 대상 지역명 출력
        print(f"Usage: {shap_chart.get('usage')}")  # 분석 대상 용도(자가용/사업자용) 출력
        print(f"Number of features: {len(shap_chart.get('features', []))}")  # 특성(feature) 개수 출력
        print(f"Number of values: {len(shap_chart.get('values', []))}")  # SHAP value(값) 개수 출력
        assert len(shap_chart['features']) == len(shap_chart['values']), "Feature and SHAP value lists must be same length!"  # 특성 수와 SHAP 값 수 일치 여부 assert(단언) 검증
        
    # --- 4. run_dr_simulation(DR 수요반응 시뮬레이션) 테스트 ---
    print("\n--- 3. Testing run_dr_simulation (V2G Peak Shaving)")  # V2G 피크 셰이빙 DR 테스트 섹션 헤더 출력
    dr_v2g_result = run_dr_simulation("안양시", "V2G_Peak_Shaving")  # V2G(차대전력망) 피크 셰이빙 시나리오 DR 시뮬레이션 실행
    sys.stdout.buffer.write(dr_v2g_result.encode('utf-8'))  # V2G DR 시뮬레이션 결과를 UTF-8로 stdout(표준출력)에 출력
    print("\n" + "-"*40)  # 섹션 구분선 출력
    
    print("\n--- 4. Testing run_dr_simulation (Smart Charging)")  # 스마트 충전 DR 테스트 섹션 헤더 출력
    dr_smart_result = run_dr_simulation("안양시", "Smart_Charging_50")  # 스마트 충전 50% 시나리오 DR 시뮬레이션 실행
    sys.stdout.buffer.write(dr_smart_result.encode('utf-8'))  # 스마트 충전 DR 결과를 UTF-8로 stdout(표준출력)에 출력
    print("\n" + "-"*40)  # 섹션 구분선 출력
    
    # --- 5. get_top_regions(상위 지역 조회) 테스트 ---
    print("\n--- 5. Testing get_top_regions (Top 10 regions by Power Load)")  # 상위 지역 조회 테스트 섹션 헤더 출력
    top_regions_result = get_top_regions(10, "전력_부하지수")  # 전력 부하지수 기준 상위 10개 지역 조회
    sys.stdout.buffer.write(top_regions_result.encode('utf-8'))  # 상위 지역 조회 결과를 UTF-8로 stdout(표준출력)에 출력
    print("\n" + "-"*40)  # 섹션 구분선 출력
    
    print("\n[SUCCESS] All functions executed without exceptions!")  # 모든 테스트 성공 메시지 출력
    print("="*60)  # 테스트 종료 구분선 출력

if __name__ == "__main__":  # 스크립트가 직접 실행될 때만 main(메인) 함수 호출
    main()  # 테스트 메인 함수 실행
