# ============================================================
# 파일명: test_v48_features.py
# 설명: V4.8 프리미엄 기능 단위 테스트(Unit Test) 스크립트.
#       TOPSIS(다기준 의사결정), 동적 요금제 시뮬레이션,
#       PDF(보고서) 생성, 통계 엔진(stats_engine) 계산을 검증한다.
# ============================================================

import sys  # sys(시스템) 모듈을 import(임포트) — 종료 코드 및 경로 제어용
import os  # os(운영체제) 모듈을 import(임포트) — 경로 조작용

# --- 프로젝트 루트(root) 경로 설정 블록 ---
# 실행 디렉토리에 관계없이 CWD(현재 작업 디렉토리)를 프로젝트 루트로 고정
script_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 테스트 스크립트 파일의 디렉터리 절대경로를 구함
project_root = os.path.abspath(os.path.join(script_dir, ".."))  # 스크립트 상위 디렉터리를 project root(프로젝트 루트)로 설정
os.chdir(project_root)  # CWD(현재 작업 디렉토리)를 프로젝트 루트로 변경
if project_root not in sys.path:  # sys.path에 프로젝트 루트가 없으면
    sys.path.insert(0, project_root)  # sys.path 최상단에 프로젝트 루트를 삽입하여 모듈 import(임포트) 가능하게 설정

import numpy as np  # numpy(넘파이) 수치 연산 라이브러리를 np로 import(임포트)
import pandas as pd  # pandas(판다스) 데이터 분석 라이브러리를 pd로 import(임포트)
from utils.optimization import calculate_topsis_rankings, simulate_dynamic_pricing  # TOPSIS(다기준 의사결정) 및 동적 요금제 시뮬레이션 함수 import(임포트)
from utils.pdf_generator import generate_regional_report_pdf  # 지역별 PDF 보고서 생성 함수 import(임포트)

# --- TOPSIS MCDA(다기준 의사결정 분석) 알고리즘 테스트 함수 ---
def test_topsis_mcda():  # TOPSIS 순위 산출 알고리즘을 mock(모의) 데이터로 검증하는 함수 정의
    print("[1/3] Testing TOPSIS MCDA Algorithm...")  # TOPSIS 테스트 시작 메시지 출력
    # --- 지역별 mock(모의) DataFrame(데이터프레임) 생성 ---
    mock_data = pd.DataFrame({  # 5개 지역의 샘플 데이터를 DataFrame(데이터프레임)으로 구성
        "지역": ["서울 강남구", "경기 안양시", "인천 남동구", "경기 수원시", "서울 종로구"],  # 행정구역 지역명 컬럼
        "용도": ["자가용", "사업자용", "자가용", "사업자용", "자가용"],  # 충전 용도 구분 컬럼
        "전력_부하지수": [80.0, 95.0, 40.0, 60.0, 50.0],  # 전력 부하지수 컬럼
        "인프라_부하지수": [90.0, 85.0, 30.0, 75.0, 65.0],  # 인프라 부하지수 컬럼
        "충전소개수": [10, 8, 3, 12, 5],  # 충전소 개수 컬럼
        "전체_충전기대수": [50, 40, 15, 60, 25],  # 전체 충전기 대수 컬럼
        "총용량_kW": [2500, 2000, 750, 3000, 1250],  # 총 충전 용량(kW) 컬럼
        "총_전력판매량": [100000, 80000, 30000, 120000, 50000]  # 총 전력 판매량 컬럼
    })
    
    weights = {  # TOPSIS 가중치(weight) dict(딕셔너리) 정의
        "전력_부하지수": 0.35,  # 전력 부하지수 가중치 35%
        "인프라_부하지수": 0.35,  # 인프라 부하지수 가중치 35%
        "충전소_밀집도_역수": 0.15,  # 충전소 밀집도 역수 가중치 15%
        "전력망_완화율": 0.15  # 전력망 완화율 가중치 15%
    }
    
    result = calculate_topsis_rankings(mock_data, weights)  # mock 데이터와 가중치로 TOPSIS 순위 계산 실행
    
    # --- 결과 컬럼 존재 여부 assert(단언) 검증 ---
    assert "TOPSIS_점수" in result.columns, "TOPSIS_점수 column missing"  # TOPSIS_점수 컬럼 존재 여부 검증
    assert "TOPSIS_순위" in result.columns, "TOPSIS_순위 column missing"  # TOPSIS_순위 컬럼 존재 여부 검증
    
    # --- 순위 범위(1~5) assert(단언) 검증 ---
    assert set(result["TOPSIS_순위"]) == {1, 2, 3, 4, 5}, "TOPSIS ranking range incorrect"  # 5개 지역의 순위가 1~5 전체를 포함하는지 검증
    print(" -> TOPSIS MCDA test: SUCCESS!")  # TOPSIS 테스트 성공 메시지 출력

# --- 동적 요금제(Dynamic Pricing) 시뮬레이션 테스트 함수 ---
def test_dynamic_pricing_simulation():  # 피크/비피크 요금 차등 시 수요 이동을 검증하는 함수 정의
    print("[2/3] Testing Dynamic Pricing Simulation...")  # 동적 요금제 테스트 시작 메시지 출력
    # --- 24시간 기준 시간대별 부하 프로파일(profile) 생성 ---
    base_profile = np.array([  # 24시간 기준 시간대별 기본 충전 부하 배열 생성
        100.0, 90.0, 80.0, 70.0, 65.0, 60.0, 70.0, 85.0, 110.0, 130.0,  # 0~9시 부하
        160.0, 170.0, 150.0, 180.0, 190.0, 185.0, 175.0, 165.0, 195.0, 210.0,  # 10~19시 부하
        205.0, 180.0, 140.0, 120.0  # 20~23시 부하
    ])
    
    elasticity = -0.2  # 수요 탄력성(elasticity) 계수: -0.2 (가격 상승 시 수요 감소)
    surcharge = 0.25  # peak surcharge(피크 할증료) 25%
    discount = 0.15  # off-peak discount(비피크 할인율) 15%
    
    sim_profile, price_change = simulate_dynamic_pricing(  # 동적 요금제 시뮬레이션 실행
        base_profile,  # 기준 부하 프로파일 전달
        elasticity=elasticity,  # 탄력성 계수 전달
        peak_surcharge=surcharge,  # 피크 할증률 전달
        discount_rate=discount  # 비피크 할인율 전달
    )
    
    # --- 출력 배열 길이 assert(단언) 검증 ---
    assert len(sim_profile) == 24, "Simulated profile must be of length 24"  # 시뮬레이션 부하 프로파일 길이가 24인지 검증
    assert len(price_change) == 24, "Price change vector must be of length 24"  # 가격 변동 벡터 길이가 24인지 검증
    
    # --- 총 수요 보존(conservation) assert(단언) 검증 ---
    original_sum = np.sum(base_profile)  # 기준 프로파일의 총 수요 합계 계산
    simulated_sum = np.sum(sim_profile)  # 시뮬레이션 후 총 수요 합계 계산
    assert np.allclose(original_sum, simulated_sum, rtol=1e-5), f"Total demand not conserved: original={original_sum}, simulated={simulated_sum}"  # 총 수요가 보존되는지 검증
    
    # --- 피크 시간대 수요 감소 assert(단언) 검증 ---
    peak_hours = [10, 11, 13, 14, 15, 16, 18, 19, 20, 21]  # 피크(peak) 시간대 인덱스 리스트 정의
    
    # surcharge(할증) 적용 시 elasticity(탄력성)<0이므로 피크 시간대 수요가 감소해야 함
    for h in peak_hours:  # 각 피크 시간대를 순회
        assert sim_profile[h] < base_profile[h], f"Demand at hour {h} did not decrease after surcharge"  # 피크 시간대 수요 감소 여부 검증
        
    print(" -> Dynamic Pricing simulation test: SUCCESS!")  # 동적 요금제 테스트 성공 메시지 출력

# --- PDF(보고서) 생성 테스트 함수 ---
def test_pdf_report_generation():  # 지역/요약/고속도로 PDF 보고서 생성을 검증하는 함수 정의
    print("[3/3] Testing Regional, Summary, and Highway PDF Report Generation...")  # PDF 테스트 시작 메시지 출력
    from utils.pdf_generator import generate_report_pdf, generate_highway_report_pdf  # 요약/고속도로 PDF 생성 함수 import(임포트)
    
    # --- mock(모의) final_data DataFrame(데이터프레임) 생성 ---
    mock_final = pd.DataFrame({  # 지역별 PDF 생성용 샘플 데이터 구성
        "지역": ["경기 안양시", "경기 안양시"],  # 지역명 컬럼
        "용도": ["자가용", "사업자용"],  # 용도 컬럼
        "전기차_전체대수": [1000.0, 800.0],  # 전기차 등록 대수 컬럼
        "전력_부하지수": [95.0, 85.0],  # 전력 부하지수 컬럼
        "인프라_부하지수": [85.0, 75.0],  # 인프라 부하지수 컬럼
        "급속충전기_대수": [20.0, 15.0],  # 급속 충전기 대수 컬럼
        "완속충전기_대수": [40.0, 30.0],  # 완속 충전기 대수 컬럼
        "전체_충전기대수": [60.0, 45.0],  # 전체 충전기 대수 컬럼
        "총용량_kW": [2000.0, 1500.0],  # 총 용량(kW) 컬럼
        "총_전력판매량": [80000.0, 60000.0],  # 총 전력 판매량 컬럼
        "총_판매수입": [200000.0, 150000.0],  # 총 판매 수입 컬럼
        "충전소개수": [10.0, 8.0]  # 충전소 개수 컬럼
    })
    
    # --- mock(모의) hourly_data DataFrame(데이터프레임) 생성 ---
    mock_hourly = pd.DataFrame({  # 시간대별 부하 데이터 샘플 구성
        "일자": ["2024-01-01", "2024-01-01"],  # 일자 컬럼
        "충전방식": ["급속", "완속"]  # 충전 방식 컬럼
    })
    for i in range(24):  # 0~23시 시간대 컬럼을 순회하며 생성
        mock_hourly[f"{i:02d}시"] = [100.0, 50.0]  # 각 시간대별 부하 값(급속 100, 완속 50) 설정
        
    # --- 1. Regional PDF(지역별 보고서) 생성 테스트 ---
    pdf_bytes_reg = generate_regional_report_pdf("안양시", mock_final, mock_hourly)  # 안양시 지역 PDF 보고서 생성
    assert isinstance(pdf_bytes_reg, (bytes, bytearray)), "Regional PDF output is not bytes"  # 출력이 bytes(바이트) 타입인지 검증
    assert pdf_bytes_reg.startswith(b"%PDF"), "Regional PDF output does not start with standard PDF header"  # PDF 표준 헤더(%PDF)로 시작하는지 검증
    print("   -> Regional PDF: SUCCESS!")  # 지역 PDF 테스트 성공 메시지 출력
    
    # --- 2. Main Summary PDF(요약 보고서) 생성 테스트 ---
    pdf_bytes_sum = generate_report_pdf(  # 메인 요약 PDF 보고서 생성
        best_name="RandomForest",  # 최적 모델명 전달
        test_rmse=0.1234,  # 테스트 RMSE(평균제곱근오차) 전달
        top3_list=["1. 서울 강남구 (자가용)", "2. 경기 안양시 (사업자용)", "3. 인천 남동구 (자가용)"],  # TOP3 지역 리스트 전달
        top_features=["total_ev_count", "infra_size_pca"],  # 상위 특성(feature) 리스트 전달
        feature_importance_img=None,  # 특성 중요도 이미지 없음
        final_data=mock_final  # mock final_data 전달
    )
    assert isinstance(pdf_bytes_sum, (bytes, bytearray)), "Summary PDF output is not bytes"  # 출력이 bytes(바이트) 타입인지 검증
    assert pdf_bytes_sum.startswith(b"%PDF"), "Summary PDF output does not start with standard PDF header"  # PDF 표준 헤더로 시작하는지 검증
    print("   -> Summary PDF: SUCCESS!")  # 요약 PDF 테스트 성공 메시지 출력
    
    # --- 3. Highway PDF(고속도로 보고서) 생성 테스트 ---
    pdf_bytes_hw = generate_highway_report_pdf(  # 고속도로 PDF 보고서 생성
        hw_df=pd.DataFrame({  # 고속도로 mock DataFrame(데이터프레임) 구성
            "unitName": ["서울IC", "안양IC"],  # 휴게소/IC 단위명 컬럼
            "routeName": ["경부선", "외곽선"],  # 노선명 컬럼
            "총용량_kW": [1000.0, 800.0],  # 총 용량(kW) 컬럼
            "부하_예측점수": [90.0, 80.0],  # 부하 예측 점수 컬럼
            "최적_추가대수": [2.0, 1.0],  # 최적 추가 충전기 대수 컬럼
            "최적화후_부하점수": [80.0, 75.0]  # 최적화 후 부하 점수 컬럼
        }),
        scenario="주말 장거리 이동 폭증",  # 교통 시나리오 전달
        budget=3  # 추가 충전기 예산 3대 전달
    )
    assert isinstance(pdf_bytes_hw, (bytes, bytearray)), "Highway PDF output is not bytes"  # 출력이 bytes(바이트) 타입인지 검증
    assert pdf_bytes_hw.startswith(b"%PDF"), "Highway PDF output does not start with standard PDF header"  # PDF 표준 헤더로 시작하는지 검증
    print("   -> Highway PDF: SUCCESS!")  # 고속도로 PDF 테스트 성공 메시지 출력
    
    print(" -> PDF report generation test: SUCCESS!")  # PDF 전체 테스트 성공 메시지 출력

# --- Stats Engine(통계 엔진) 계산 테스트 함수 ---
def test_stats_engine_computations():  # stats_engine 모듈의 핵심 통계 함수들을 검증하는 함수 정의
    print("[4/4] Testing Stats Engine Computations...")  # 통계 엔진 테스트 시작 메시지 출력
    from utils.stats_engine import (  # stats_engine(통계 엔진) 모듈에서 계산 함수들 import(임포트)
        calculate_carbon_offset,  # CO2(이산화탄소) 저감량 계산 함수
        calculate_fiscal_roi_metrics,  # 재정 ROI(투자수익률) 지표 계산 함수
        calculate_solar_absorption_kwh,  # 태양광 흡수 잠재량 계산 함수
        calculate_avg_capacity_per_charger  # 충전기 대당 평균 용량 계산 함수
    )
    
    mock_final = pd.DataFrame({  # 통계 계산용 mock DataFrame(데이터프레임) 생성
        "지역": ["경기 안양시", "경기 부천시", "경기 안성시"],  # 지역명 컬럼
        "용도": ["자가용", "사업자용", "사업자용"],  # 용도 컬럼
        "전기차_전체대수": [1000.0, 800.0, 500.0],  # 전기차 등록 대수 컬럼
        "전력_부하지수": [95.0, 140.0, 85.0],  # 전력 부하지수 컬럼
        "인프라_부하지수": [85.0, 42.0, 10.0],  # 인프라 부하지수 컬럼
        "급속충전기_대수": [20.0, 15.0, 10.0],  # 급속 충전기 대수 컬럼
        "완속충전기_대수": [40.0, 30.0, 20.0],  # 완속 충전기 대수 컬럼
        "전체_충전기대수": [60.0, 45.0, 30.0],  # 전체 충전기 대수 컬럼
        "총용량_kW": [2000.0, 1500.0, 1000.0],  # 총 용량(kW) 컬럼
        "총_전력판매량": [80000.0, 120000.0, 60000.0],  # 총 전력 판매량 컬럼
        "총_판매수입": [200000.0, 300000.0, 150000.0]  # 총 판매 수입 컬럼
    })
    
    # --- 1. Carbon Offset(CO2 저감) 계산 테스트 ---
    offset = calculate_carbon_offset(100000.0)  # 총 판매량 100,000 kWh 기준 CO2 저감량 계산
    assert offset > 0, "Carbon offset must be positive"  # CO2 저감량이 양수인지 검증
    
    # --- 2. Fiscal ROI(재정 투자수익률) 계산 테스트 ---
    roi_mul, budget_saving = calculate_fiscal_roi_metrics(mock_final)  # mock 데이터로 ROI 배수 및 예산 절감율 계산
    assert roi_mul > 0, "ROI multiplier must be positive"  # ROI 배수가 양수인지 검증
    assert 15.0 <= budget_saving <= 45.0, "Budget saving rate must be bounded"  # 예산 절감율이 15~45% 범위 내인지 검증
    
    # --- 3. Solar Absorption(태양광 흡수) 계산 테스트 ---
    solar = calculate_solar_absorption_kwh(100000.0)  # 총 판매량 100,000 kWh 기준 태양광 흡수 잠재량 계산
    assert solar == 8000.0, "Solar absorption should be 8% of sales"  # 태양광 흡수량이 판매량의 8%(8,000 kWh)인지 검증
    
    # --- 4. Avg Capacity(평균 용량) 계산 테스트 ---
    avg_cap = calculate_avg_capacity_per_charger(mock_final)  # 충전기 대당 평균 용량 계산
    assert np.isclose(avg_cap, 4500.0 / 135.0), "Avg capacity calculation incorrect"  # 평균 용량이 4500/135와 근사한지 검증
    
    print(" -> Stats Engine calculations test: SUCCESS!")  # 통계 엔진 테스트 성공 메시지 출력

if __name__ == "__main__":  # 스크립트가 직접 실행될 때만 테스트 실행
    print("==================================================")  # 테스트 시작 구분선 출력
    print("RUNNING V4.8 PREMIUM FEATURES UNIT TESTS...")  # V4.8 테스트 제목 출력
    print("==================================================")  # 구분선 출력
    
    try:  # 테스트 실행 중 예외 처리를 위한 try 블록 시작
        test_topsis_mcda()  # TOPSIS MCDA 테스트 실행
        test_dynamic_pricing_simulation()  # 동적 요금제 시뮬레이션 테스트 실행
        test_pdf_report_generation()  # PDF 보고서 생성 테스트 실행
        test_stats_engine_computations()  # 통계 엔진 계산 테스트 실행
        print("\n -> ALL TESTS PASSED SUCCESSFULLY!")  # 전체 테스트 성공 메시지 출력
        sys.exit(0)  # exit code(종료 코드) 0으로 정상 종료
    except AssertionError as ae:  # assert(단언) 실패 예외 포착
        print(f"\n -> Assertion failed: {ae}")  # 단언 실패 메시지 출력
        sys.exit(1)  # exit code(종료 코드) 1로 실패 종료
    except Exception as e:  # 기타 모든 예외 포착
        import traceback  # traceback(트레이스백) 모듈을 import(임포트)하여 상세 오류 출력
        traceback.print_exc()  # 전체 스택 트레이스를 콘솔에 출력
        print(f"\n -> Test execution encountered error: {e}")  # 오류 메시지 출력
        sys.exit(1)  # exit code(종료 코드) 1로 실패 종료
