import streamlit as st
import os
import sys

# 실행 디렉토리에 관계없이 작업 디렉토리(CWD)를 프로젝트 루트로 강제 고정하고 sys.path 등록
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.data_processing import load_all_data, load_precomputed_analytics
from views.ai_assistant import get_simulation_result, get_shap_analysis, run_dr_simulation, get_top_regions

def main():
    print("="*60)
    print("Running AI Assistant Features Unit Test (SHAP & DR Sim)")
    print("="*60)
    
    # 1. 데이터 로드 및 세션 상태 설정
    dataset_dir = "./dataset"
    print(f"Loading dataset from: {dataset_dir}")
    final_data, monthly_data, hourly_data = load_all_data(dataset_dir)
    st.session_state["final_data"] = final_data
    
    print("Columns in final_data:")
    columns_str = ", ".join(final_data.columns.tolist())
    sys.stdout.buffer.write((columns_str + "\n").encode('utf-8'))
    
    print("Unique regions containing '안양':")
    anyang_regions = final_data[final_data["지역"].str.contains("안양", na=False)]["지역"].unique().tolist()
    sys.stdout.buffer.write((", ".join(anyang_regions) + "\n").encode('utf-8'))
    
    json_path = os.path.join("results", "precomputed_analytics.json")
    onnx_path = os.path.join("results", "best_model.onnx")
    
    print(f"Loading precomputed model states: {json_path}, {onnx_path}")
    model_state, model_state_smote = load_precomputed_analytics(json_path, onnx_path)
    st.session_state["model_state"] = model_state
    
    # 2. get_simulation_result 테스트
    print("\n--- 1. Testing get_simulation_result ---")
    sim_result = get_simulation_result("안양시", 10, "급속")
    sys.stdout.buffer.write(sim_result.encode('utf-8'))
    print("\n" + "-"*40)
    
    # 3. get_shap_analysis 테스트
    print("\n--- 2. Testing get_shap_analysis ---")
    shap_result = get_shap_analysis("안양시")
    sys.stdout.buffer.write(shap_result.encode('utf-8'))
    print("\n" + "-"*40)
    
    # 세션 상태 체크
    shap_chart = st.session_state.get("last_shap_chart")
    print(f"last_shap_chart stored in st.session_state: {shap_chart is not None}")
    if shap_chart:
        print(f"Keys: {list(shap_chart.keys())}")
        print(f"Region: {shap_chart.get('region')}")
        print(f"Usage: {shap_chart.get('usage')}")
        print(f"Number of features: {len(shap_chart.get('features', []))}")
        print(f"Number of values: {len(shap_chart.get('values', []))}")
        assert len(shap_chart['features']) == len(shap_chart['values']), "Feature and SHAP value lists must be same length!"
        
    # 4. run_dr_simulation 테스트
    print("\n--- 3. Testing run_dr_simulation (V2G Peak Shaving) ---")
    dr_v2g_result = run_dr_simulation("안양시", "V2G_Peak_Shaving")
    sys.stdout.buffer.write(dr_v2g_result.encode('utf-8'))
    print("\n" + "-"*40)
    
    print("\n--- 4. Testing run_dr_simulation (Smart Charging) ---")
    dr_smart_result = run_dr_simulation("안양시", "Smart_Charging_50")
    sys.stdout.buffer.write(dr_smart_result.encode('utf-8'))
    print("\n" + "-"*40)
    
    # 5. get_top_regions 테스트
    print("\n--- 5. Testing get_top_regions (Top 10 regions by Power Load) ---")
    top_regions_result = get_top_regions(10, "전력_부하지수")
    sys.stdout.buffer.write(top_regions_result.encode('utf-8'))
    print("\n" + "-"*40)
    
    print("\n[SUCCESS] All functions executed without exceptions!")
    print("="*60)

if __name__ == "__main__":
    main()
