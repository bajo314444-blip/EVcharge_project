# ============================================================
# 파일명: stats_engine.py
# 설명: 통계 분석 엔진 모듈.
#       CO2 저감 효과 산출, 재정 ROI(투자수익률) 지표 계산,
#       태양광 잉여 전력 흡수량 추정, 충전기 대당 평균 용량 계산 등
#       정책 보고서에 필요한 핵심 통계 함수들을 제공한다.
# ============================================================

import pandas as pd  # pandas(판다스) 데이터 분석 라이브러리를 pd로 import(임포트)
import numpy as np  # numpy(넘파이) 수치 연산 라이브러리를 np로 import(임포트)

# --- CO2(이산화탄소) 저감 효과 산출 함수 ---
def calculate_carbon_offset(total_sales_kwh, peak_shift_rate=0.125):  # 총 판매 전력량(kWh)과 피크 이전율을 받아 CO2 저감량을 계산하는 함수 정의
    """
    피크 시간대 충전 부하의 분산 이전량 대비 온실가스(CO2) 저감 효과를 톤 단위로 산출합니다.
    - 배출계수: 0.457 kg CO2 / kWh (대한민국 전력 평균 배출량 표준)
    """
    peak_shaved_kwh = total_sales_kwh * peak_shift_rate  # 총 판매 전력량에 피크 이전율을 곱하여 피크 절감 전력량(kWh) 산출
    co2_reduction_kg = peak_shaved_kwh * 0.457  # 피크 절감 전력량에 배출계수(0.457 kg CO2/kWh)를 곱하여 CO2 감소량(kg) 산출
    return co2_reduction_kg / 1000.0  # kg 단위를 톤(Tons) 단위로 변환하여 반환

# --- 재정 ROI(투자수익률) 지표 계산 함수 ---
def calculate_fiscal_roi_metrics(final_data):  # 전체 데이터를 받아 ROI 배수와 예산 절감율을 계산하는 함수 정의
    """
    전체 수도권 매출 데이터 대비 전력 부하지수 최우수 지역(TOPSIS 상위 3대 취약 구역)의 
    대당 매출 효율을 계산하여 투자 회수 속도(배수) 및 예산 낭비 절감율을 동적 반환합니다.
    """
    if final_data is None or final_data.empty:  # 입력 데이터가 None이거나 비어있는 경우 확인
        return 1.4, 28.0  # 기본 fallback(폴백) 값 (ROI 배수 1.4, 예산 절감율 28.0%)을 반환

    try:  # 계산 중 예외 발생 시 처리하기 위한 try 블록 시작
        # --- TOPSIS(토프시스) 상위 3순위를 대변하는 전력 부하지수 상위 3개 지역 추출 ---
        top3_df = final_data.sort_values("전력_부하지수", ascending=False).head(3)  # 전력 부하지수 기준 내림차순 정렬 후 상위 3개 행 추출

        sum_revenue_top3 = top3_df["총_판매수입"].sum()  # 상위 3개 지역의 총 판매수입 합계 산출
        sum_chargers_top3 = top3_df["전체_충전기대수"].sum()  # 상위 3개 지역의 전체 충전기 대수 합계 산출

        sum_revenue_all = final_data["총_판매수입"].sum()  # 전체 데이터의 총 판매수입 합계 산출
        sum_chargers_all = final_data["전체_충전기대수"].sum()  # 전체 데이터의 전체 충전기 대수 합계 산출

        avg_revenue_top3 = sum_revenue_top3 / sum_chargers_top3 if sum_chargers_top3 > 0 else 0  # 상위 3개 지역의 충전기 대당 평균 매출 계산 (0 나눗셈 방지)
        avg_revenue_all = sum_revenue_all / sum_chargers_all if sum_chargers_all > 0 else 0  # 전체 지역의 충전기 대당 평균 매출 계산 (0 나눗셈 방지)

        roi_multiplier = avg_revenue_top3 / avg_revenue_all if avg_revenue_all > 0 else 1.4  # 상위 대비 전체 평균 매출 비율로 ROI 배수 산출 (0 나눗셈 방지)

        # --- 최하위 순위 지역과의 매출 대비 효율 계산을 통해 예산 절감 가능율 산출 ---
        lowest_df = final_data.sort_values("전력_부하지수", ascending=True).head(10)  # 전력 부하지수 기준 오름차순 정렬 후 하위 10개 행 추출
        avg_revenue_lowest = lowest_df["총_판매수입"].sum() / lowest_df["전체_충전기대수"].sum() if lowest_df["전체_충전기대수"].sum() > 0 else 0  # 하위 10개 지역의 충전기 대당 평균 매출 계산

        if avg_revenue_top3 > 0:  # 상위 3개 지역의 평균 매출이 0보다 큰 경우
            budget_saving_rate = ((avg_revenue_top3 - avg_revenue_lowest) / avg_revenue_top3) * 100  # 상위와 하위 간 매출 차이를 백분율로 환산하여 예산 절감율 산출
            # 최소 15%, 최대 45% 범위 내에서 보정
            budget_saving_rate = np.clip(budget_saving_rate, 15.0, 45.0)  # 예산 절감율을 15~45% 범위로 clip(클리핑)하여 제한
        else:  # 상위 3개 지역의 평균 매출이 0 이하인 경우
            budget_saving_rate = 28.0  # 기본 fallback(폴백) 예산 절감율 28.0% 적용

        return float(roi_multiplier), float(budget_saving_rate)  # ROI 배수와 예산 절감율을 float(실수) 타입으로 변환하여 반환
    except Exception:  # 계산 과정에서 발생하는 모든 예외를 포착
        return 1.4, 28.0  # 예외 발생 시 기본 fallback(폴백) 값 반환

# --- 태양광 잉여 전력 흡수량 추정 함수 ---
def calculate_solar_absorption_kwh(total_sales_kwh, shift_rate=0.08):  # 총 판매 전력량과 이전율을 받아 태양광 흡수 잠재량을 계산하는 함수 정의
    """
    다이내믹 요금제의 주간(12~15시) 특별 할인 도입을 통해 
    전력망 내 태양광 잉여 발전 출력을 흡수 및 상쇄할 수 있는 잠재량을 kWh 단위로 도출합니다.
    """
    return total_sales_kwh * shift_rate  # 총 판매 전력량에 이전율을 곱하여 태양광 흡수 잠재량(kWh) 반환

# --- 충전기 대당 평균 용량 계산 함수 ---
def calculate_avg_capacity_per_charger(matched_df):  # 특정 지역의 DataFrame(데이터프레임)을 받아 충전기 대당 평균 용량을 계산하는 함수 정의
    """
    특정 행정구역(또는 자치구)의 충전기 1대당 평균 용량(kW)을 계산합니다.
    """
    if matched_df is None or matched_df.empty:  # 입력 데이터가 None이거나 비어있는 경우 확인
        return 50.0  # 기본 fallback(폴백) 값 50.0 kW를 반환
    total_kw = matched_df["총용량_kW"].sum()  # 해당 지역의 총 용량(kW) 합계 산출
    total_chargers = matched_df["전체_충전기대수"].sum()  # 해당 지역의 전체 충전기 대수 합계 산출
    return total_kw / total_chargers if total_chargers > 0 else 50.0  # 대당 평균 용량 계산 (0 나눗셈 방지, 기본값 50.0 kW)
