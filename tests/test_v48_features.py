import sys
import os

# 실행 디렉토리에 관계없이 CWD를 프로젝트 루트로 고정
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from utils.optimization import calculate_topsis_rankings, simulate_dynamic_pricing
from utils.pdf_generator import generate_regional_report_pdf

def test_topsis_mcda():
    print("[1/3] Testing TOPSIS MCDA Algorithm...")
    # Create mock DataFrame representing regions
    mock_data = pd.DataFrame({
        "지역": ["서울 강남구", "경기 안양시", "인천 남동구", "경기 수원시", "서울 종로구"],
        "용도": ["자가용", "사업자용", "자가용", "사업자용", "자가용"],
        "전력_부하지수": [80.0, 95.0, 40.0, 60.0, 50.0],
        "인프라_부하지수": [90.0, 85.0, 30.0, 75.0, 65.0],
        "충전소개수": [10, 8, 3, 12, 5],
        "전체_충전기대수": [50, 40, 15, 60, 25],
        "총용량_kW": [2500, 2000, 750, 3000, 1250],
        "총_전력판매량": [100000, 80000, 30000, 120000, 50000]
    })
    
    weights = {
        "전력_부하지수": 0.35,
        "인프라_부하지수": 0.35,
        "충전소_밀집도_역수": 0.15,
        "전력망_완화율": 0.15
    }
    
    result = calculate_topsis_rankings(mock_data, weights)
    
    # Assert columns exist
    assert "TOPSIS_점수" in result.columns, "TOPSIS_점수 column missing"
    assert "TOPSIS_순위" in result.columns, "TOPSIS_순위 column missing"
    
    # Assert rankings are 1 to 5
    assert set(result["TOPSIS_순위"]) == {1, 2, 3, 4, 5}, "TOPSIS ranking range incorrect"
    print(" -> TOPSIS MCDA test: SUCCESS!")

def test_dynamic_pricing_simulation():
    print("[2/3] Testing Dynamic Pricing Simulation...")
    # 24 hours of base hourly load profile
    base_profile = np.array([
        100.0, 90.0, 80.0, 70.0, 65.0, 60.0, 70.0, 85.0, 110.0, 130.0, 
        160.0, 170.0, 150.0, 180.0, 190.0, 185.0, 175.0, 165.0, 195.0, 210.0,
        205.0, 180.0, 140.0, 120.0
    ])
    
    elasticity = -0.2
    surcharge = 0.25 # 25% peak surcharge
    discount = 0.15  # 15% off-peak discount
    
    sim_profile, price_change = simulate_dynamic_pricing(
        base_profile,
        elasticity=elasticity,
        peak_surcharge=surcharge,
        discount_rate=discount
    )
    
    # Assert shape is 24
    assert len(sim_profile) == 24, "Simulated profile must be of length 24"
    assert len(price_change) == 24, "Price change vector must be of length 24"
    
    # Assert demand conservation (within float precision tolerances)
    original_sum = np.sum(base_profile)
    simulated_sum = np.sum(sim_profile)
    assert np.allclose(original_sum, simulated_sum, rtol=1e-5), f"Total demand not conserved: original={original_sum}, simulated={simulated_sum}"
    
    # Assert peak hours decreased and off-peak hours increased
    peak_hours = [10, 11, 13, 14, 15, 16, 18, 19, 20, 21]
    
    # Surcharge applied to peak hours, so demand should drop (since elasticity < 0 and surcharge > 0)
    for h in peak_hours:
        assert sim_profile[h] < base_profile[h], f"Demand at hour {h} did not decrease after surcharge"
        
    print(" -> Dynamic Pricing simulation test: SUCCESS!")

def test_pdf_report_generation():
    print("[3/3] Testing Regional, Summary, and Highway PDF Report Generation...")
    from utils.pdf_generator import generate_report_pdf, generate_highway_report_pdf
    
    # Create mock final_data
    mock_final = pd.DataFrame({
        "지역": ["경기 안양시", "경기 안양시"],
        "용도": ["자가용", "사업자용"],
        "전기차_전체대수": [1000.0, 800.0],
        "전력_부하지수": [95.0, 85.0],
        "인프라_부하지수": [85.0, 75.0],
        "급속충전기_대수": [20.0, 15.0],
        "완속충전기_대수": [40.0, 30.0],
        "전체_충전기대수": [60.0, 45.0],
        "총용량_kW": [2000.0, 1500.0],
        "총_전력판매량": [80000.0, 60000.0],
        "총_판매수입": [200000.0, 150000.0],
        "충전소개수": [10.0, 8.0]
    })
    
    # Create mock hourly_data
    mock_hourly = pd.DataFrame({
        "일자": ["2024-01-01", "2024-01-01"],
        "충전방식": ["급속", "완속"]
    })
    for i in range(24):
        mock_hourly[f"{i:02d}시"] = [100.0, 50.0]
        
    # 1. Regional PDF
    pdf_bytes_reg = generate_regional_report_pdf("안양시", mock_final, mock_hourly)
    assert isinstance(pdf_bytes_reg, (bytes, bytearray)), "Regional PDF output is not bytes"
    assert pdf_bytes_reg.startswith(b"%PDF"), "Regional PDF output does not start with standard PDF header"
    print("   -> Regional PDF: SUCCESS!")
    
    # 2. Main Summary PDF (with TOP 10 Table)
    pdf_bytes_sum = generate_report_pdf(
        best_name="RandomForest",
        test_rmse=0.1234,
        top3_list=["1. 서울 강남구 (자가용)", "2. 경기 안양시 (사업자용)", "3. 인천 남동구 (자가용)"],
        top_features=["total_ev_count", "infra_size_pca"],
        feature_importance_img=None,
        final_data=mock_final
    )
    assert isinstance(pdf_bytes_sum, (bytes, bytearray)), "Summary PDF output is not bytes"
    assert pdf_bytes_sum.startswith(b"%PDF"), "Summary PDF output does not start with standard PDF header"
    print("   -> Summary PDF: SUCCESS!")
    
    # 3. Highway PDF
    pdf_bytes_hw = generate_highway_report_pdf(
        hw_df=pd.DataFrame({
            "unitName": ["서울IC", "안양IC"],
            "routeName": ["경부선", "외곽선"],
            "총용량_kW": [1000.0, 800.0],
            "부하_예측점수": [90.0, 80.0],
            "최적_추가대수": [2.0, 1.0],
            "최적화후_부하점수": [80.0, 75.0]
        }),
        scenario="주말 장거리 이동 폭증",
        budget=3
    )
    assert isinstance(pdf_bytes_hw, (bytes, bytearray)), "Highway PDF output is not bytes"
    assert pdf_bytes_hw.startswith(b"%PDF"), "Highway PDF output does not start with standard PDF header"
    print("   -> Highway PDF: SUCCESS!")
    
    print(" -> PDF report generation test: SUCCESS!")

if __name__ == "__main__":
    print("==================================================")
    print("RUNNING V4.8 PREMIUM FEATURES UNIT TESTS...")
    print("==================================================")
    
    try:
        test_topsis_mcda()
        test_dynamic_pricing_simulation()
        test_pdf_report_generation()
        print("\n -> ALL TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    except AssertionError as ae:
        print(f"\n -> Assertion failed: {ae}")
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n -> Test execution encountered error: {e}")
        sys.exit(1)
