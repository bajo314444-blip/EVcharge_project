import sys
import os

# 실행 디렉토리에 관계없이 작업 디렉토리(CWD)를 프로젝트 루트로 강제 고정하고 sys.path 등록
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import pandas as pd
import numpy as np

def run_health_check():
    print("==================================================")
    print("STARTING TWO-TRACK SYSTEM HEALTH CHECK (V4.3)...")
    print("==================================================")
    
    # 1. Test imports
    print("[1/6] Testing imports...")
    try:
        from utils.data_processing import load_all_data, load_highway_data, DEFAULT_DATA_DIR, load_precomputed_analytics, ONNXModelWrapper
        from utils.models import train_models, make_feature_matrix
        from utils.optimization import optimize_highway_chargers, calculate_single_region_trajectory
        from utils.visualizations import make_bubble_map, make_tableone, render_shap_or_fallback, render_highway_edge_plot
        from utils.pdf_generator import generate_report_pdf, generate_highway_report_pdf
        from views.urban_dashboard import render_dashboard, render_report
        from views.highway_dashboard import render_highway_dashboard
        from views.ai_assistant import render_ai_assistant
        print(" -> All module imports: SUCCESS!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f" -> Import failed: {e}")
        sys.exit(1)
        
    # 2. Test data loading
    print("[2/6] Testing data loading from dataset/...")
    try:
        final_data, monthly_data, hourly_data = load_all_data(DEFAULT_DATA_DIR)
        print(f" -> Urban data: {len(final_data)} rows loaded.")
        print(f" -> Monthly data: {len(monthly_data)} rows loaded.")
        
        hw_data = load_highway_data(DEFAULT_DATA_DIR)
        print(f" -> Highway data: {len(hw_data)} rows loaded.")
        print(" -> Data loading: SUCCESS!")
    except Exception as e:
        print(f" -> Data loading failed: {e}")
        sys.exit(1)
        
    # 3. Test ONNX and JSON two-track loading
    print("[3/6] Testing Two-track (ONNX + JSON) loader...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, "results", "precomputed_analytics.json")
    onnx_path = os.path.join(base_dir, "results", "best_model.onnx")
    
    if not os.path.exists(json_path) or not os.path.exists(onnx_path):
        print(f" -> Required model files not found at {json_path} or {onnx_path}! Please run save_models.py first.")
        sys.exit(1)
        
    try:
        model_state, model_state_smote = load_precomputed_analytics(json_path, onnx_path)
        print(" -> Loaded precomputed analytics and ONNX model successfully.")
        print(f" -> Best model: {model_state['best_name']}")
        print(f" -> Feature columns: {model_state['feature_columns']}")
        print(f" -> Metrics shape: {model_state['metrics'].shape}")
        print(f" -> Importance shape: {model_state['importance'].shape}")
        if model_state_smote:
            print(f" -> SMOTE metrics shape: {model_state_smote['metrics'].shape}")
        print(" -> Two-track loading: SUCCESS!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f" -> Two-track loading failed: {e}")
        sys.exit(1)
        
    # 4. Test predictions with ONNX Wrapper
    print("[4/6] Testing prediction logic using ONNX runtime wrapper...")
    try:
        best_name = model_state["best_name"]
        best_model = model_state["models"][best_name]
        X_test = model_state["X_test"]
        
        # Test shape conversion
        preds = best_model.predict(X_test)
        print(f" -> Batch predictions shape: {preds.shape} (Test set size: {len(X_test)})")
        print(f" -> Batch predictions sample: {preds[:5]}")
        
        # Test single-row conversion
        single_row = X_test.iloc[0]
        single_pred = best_model.predict(single_row)
        print(f" -> Single prediction output: {single_pred}")
        
        print(" -> ONNX Inference: SUCCESS!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f" -> ONNX Prediction failed: {e}")
        sys.exit(1)
        
    # 5. Test optimization algorithms
    print("[5/6] Testing highway location optimization...")
    try:
        if not hw_data.empty:
            hw_sim = hw_data.copy()
            hw_sim["부하_예측점수"] = 50.0
            hw_optimized = optimize_highway_chargers(hw_sim, budget=10)
            print(f" -> Highway optimization succeeded. Added chargers: {hw_optimized['최적_추가대수'].sum()}")
        print(" -> Optimization: SUCCESS!")
    except Exception as e:
        print(f" -> Optimization failed: {e}")
        sys.exit(1)
        
    # 6. Test FPDF PDF report generation
    print("[6/6] Testing PDF generation logic...")
    try:
        import tempfile
        tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_img_name = tmp_img.name
        tmp_img.close()
        
        # Save a dummy white image for testing
        from PIL import Image
        img = Image.new('RGB', (100, 100), color = 'white')
        img.save(tmp_img_name)
        
        # Generate urban report
        pdf_bytes = generate_report_pdf(
            best_name=model_state["best_name"],
            test_rmse=0.5,
            top3_list=["1. Seoul Gangnam - Load 0.95", "2. Gyeonggi Anyang - Load 0.85"],
            top_features=["total_ev_count", "infra_size_pca"],
            feature_importance_img=tmp_img_name
        )
        print(f" -> Urban PDF report generated: {len(pdf_bytes)} bytes.")
        
        # Clean up
        os.remove(tmp_img_name)
        print(" -> PDF report generation: SUCCESS!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f" -> PDF generation failed: {e}")
        sys.exit(1)
        
    print("==================================================")
    print("SYSTEM HEALTH CHECK: ALL PASSED!")
    print("==================================================")

if __name__ == "__main__":
    run_health_check()
