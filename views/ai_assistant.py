# ============================================================
# 파일명: ai_assistant.py
# 설명: AI 관제비서 뷰 모듈.
#       Google Gemini(제미니) 및 Groq(그록) LLM(대규모 언어 모델) 연동,
#       Function Calling(함수 호출) 기반 시뮬레이션·SHAP·PDF 보고서 생성.
# ============================================================

import streamlit as st  # streamlit(스트림릿) 웹 앱 프레임워크를 st로 import(임포트)
import google.generativeai as genai  # google.generativeai(제미니 AI SDK)를 genai로 import(임포트)
import json  # json(JSON) 직렬화/역직렬화 모듈 import(임포트)
import gc  # gc(가비지 컬렉션) 모듈 import(임포트) — 메모리 즉시 반환용

# --- 충전기 증설 시뮬레이션 Function Calling(함수 호출) 대상 함수 ---
@st.cache_data(show_spinner=False, ttl=600)  # cache_data(데이터 캐시) 데코레이터 — 600초 TTL(유효시간) 캐싱
def get_simulation_result(region: str, count: int, power_type: str = "급속") -> str:  # 지역·대수·타입으로 증설 시뮬레이션 결과 반환
    """
    지정된 수도권 행정구역에 특정 충전기 타입과 대수를 증설했을 때의 
    예측 전력 부하지수 감소량 및 전/후 비교 결과를 조회하거나 실시간 계산합니다.
    
    Args:
        region: 분석할 행정구역명 (예: '안양시 동안구', '안양시', '수원시 동안구' 등)
        count: 설치할 충전기 대수 (예: 10)
        power_type: 충전기 타입 ('급속' 또는 '완속')
    Returns:
        시뮬레이션 전/후 비교 수치가 담긴 상세 요약 텍스트
    """
    import numpy as np  # numpy(넘파이) 수치 연산 라이브러리 import(임포트)
    import pandas as pd  # pandas(판다스) DataFrame(데이터프레임) 처리 라이브러리 import(임포트)
    from utils.optimization import calculate_single_region_trajectory  # 단일 지역 trajectory(궤적) 계산 함수 import(임포트)

    final_data = st.session_state.get("final_data")  # session_state(세션 상태)에서 최종 분석 데이터 조회
    if final_data is None:  # 데이터가 없으면
        return "시스템 에러: 데이터가 세션에 존재하지 않습니다."  # 에러 메시지 반환

    # --- 1. 지역명 매칭: "지역" 컬럼 → "시군구" 컬럼 순으로 fuzzy match(퍼지 매칭) ---
    matched = final_data[final_data["지역"].str.contains(region, case=False, na=False)]  # "지역" 컬럼에서 region(지역명) 부분 일치 검색
    if matched.empty:  # "지역" 컬럼에서 매칭 실패 시
        matched = final_data[final_data["시군구"].str.contains(region, case=False, na=False)]  # "시군구" 컬럼으로 재검색
        if matched.empty:  # 두 컬럼 모두 매칭 실패 시
            return f"검색된 지역 '{region}'을 찾을 수 없습니다. 경기 안양시 동안구 등 수도권 내 정확한 행정구역명을 입력해 주세요."  # 안내 메시지 반환

    # --- 2. 용량 매핑: 급속 100kW, 완속 7kW 기준으로 증설 kW(킬로와트) 계산 ---
    added_kw = count * (100.0 if power_type == "급속" else 7.0)  # count(대수) × 타입별 단위 용량 = added_kw(추가 용량)
    critical_threshold = float(final_data["전력_부하지수"].quantile(0.8))  # quantile(분위수) 80% = 고위험 임계치

    lines = []  # 마크다운 결과 줄 리스트 초기화
    lines.append(f"### 🔮 {region} 지역 충전기 증설 시뮬레이션 결과")  # 제목 줄 추가
    lines.append(f"- **증설 정책**: {power_type} 충전소 {count}대 증설 (공급 용량: +{added_kw:,.1f} kW)")  # 정책 요약 줄 추가
    lines.append(f"- **전력 부하 임계치**: {critical_threshold:,.2f} (상위 20% 고위험 기준선)\n")  # 임계치 줄 추가

    # --- 3. 매칭된 각 세부 구군별 시뮬레이션 루프 ---
    for idx, row in matched.iterrows():  # matched(매칭된) DataFrame(데이터프레임) 행 순회
        r_name = row["지역"]  # 행정구역명 추출
        usage = row["용도"]  # 용도(자가용/사업자용) 추출
        before_load = float(row["전력_부하지수"])  # 증설 전 전력 부하지수
        base_load = float(row["총_전력판매량"])  # 총 전력판매량(기준 부하)
        capacity = float(row["총용량_kW"])  # 현재 충전기 총용량(kW)

        after_load = base_load / (capacity + added_kw) if (capacity + added_kw) > 0 else 0  # 증설 후 부하지수 = 판매량 / (기존+추가 용량)
        reduction_pct = (before_load - after_load) / before_load * 100 if before_load > 0 else 0  # reduction_pct(감소율 %) 계산

        # --- 생존 분석: 미래 과부하 도달 시점 예측 (기본 성장률 5%) ---
        traj_df, overload_before, overload_after = calculate_single_region_trajectory(  # trajectory(궤적) 함수 호출
            base_load, capacity, 0.05, added_kw, critical_threshold  # base_load, capacity, growth(성장률), added_kw, threshold(임계치)
        )
        
        delay_years = overload_after - overload_before  # delay_years(과부하 지연 연수) = 증설 후 − 증설 전
        delay_text = f"+{delay_years}년 지연" if delay_years > 0 else "변동 없음"  # 지연 효과 텍스트 생성
        before_txt = f"{overload_before}년 뒤" if overload_before < 15 else "안전 (15년+)"  # 증설 전 과부하 도달 시점 텍스트
        after_txt = f"{overload_after}년 뒤" if overload_after < 15 else "안전 (15년+)"  # 증설 후 과부하 도달 시점 텍스트

        lines.append(f"#### 📍 {r_name} ({usage} 기준)")  # 구군별 소제목 줄 추가
        lines.append(f"  - **전력 부하지수**: {before_load:,.2f} ➡️ **{after_load:,.2f}** ({reduction_pct:+.1f}% 감소)")  # 부하지수 전/후 비교 줄
        lines.append(f"  - **과부하 도달 시점**: {before_txt} ➡️ **{after_txt}** (지연 효과: {delay_text})")  # 과부하 시점 비교 줄
        lines.append("")  # 구분용 빈 줄 추가

    gc.collect()  # gc.collect(가비지 컬렉션) 강제 실행 — 캐시 함수 메모리 즉시 반환

    return "\n".join(lines)  # lines(줄 리스트)를 개행으로 join(결합)하여 마크다운 문자열 반환

# --- SHAP(설명 가능 AI) 기여도 분석 Function Calling(함수 호출) 대상 함수 ---
@st.cache_data(show_spinner=False, ttl=600)  # cache_data(데이터 캐시) 데코레이터 — 600초 TTL(유효시간) 캐싱
def get_shap_analysis(region: str) -> str:  # 지정 지역의 SHAP(설명 가능 AI) 기여도 분석 결과 반환
    """
    지정된 수도권 행정구역의 부하 예측 결과에 대해 
    각 피처(인프라, 전기차수, 용량 등)가 미친 긍정/부정적 기여도(SHAP)를 분석합니다.
    
    Args:
        region: 분석할 행정구역명 (예: '안양시 동안구', '안양시', '수원시 동안구' 등)
    Returns:
        기여도가 높은 주요 변수 분석 및 기여도 요약 텍스트
    """
    import numpy as np  # numpy(넘파이) 수치 연산 라이브러리 import(임포트)
    import pandas as pd  # pandas(판다스) DataFrame(데이터프레임) 처리 라이브러리 import(임포트)
    import gc  # gc(가비지 컬렉션) 모듈 import(임포트)
    from utils.visualizations import get_cached_local_shap  # 캐시된 local SHAP(로컬 설명) 계산 함수 import(임포트)
    
    final_data = st.session_state.get("final_data")  # session_state(세션 상태)에서 최종 분석 데이터 조회
    model_state = st.session_state.get("model_state")  # session_state(세션 상태)에서 ML(머신러닝) 모델 상태 조회
    if final_data is None or model_state is None:  # 데이터 또는 모델 상태가 없으면
        return "시스템 에러: 모델 상태 또는 데이터가 존재하지 않습니다."  # 에러 메시지 반환
        
    # --- 지역명 매칭: "지역" → "시군구" 순으로 fuzzy match(퍼지 매칭) ---
    matched = final_data[final_data["지역"].str.contains(region, case=False, na=False)]  # "지역" 컬럼 부분 일치 검색
    if matched.empty:  # "지역" 컬럼 매칭 실패 시
        matched = final_data[final_data["시군구"].str.contains(region, case=False, na=False)]  # "시군구" 컬럼으로 재검색
        if matched.empty:  # 두 컬럼 모두 매칭 실패 시
            return f"검색된 지역 '{region}'을 찾을 수 없습니다. 경기 안양시 동안구 등 수도권 내 정확한 행정구역명을 입력해 주세요."  # 안내 메시지 반환
            
    best_model = model_state["models"][model_state["best_name"]]  # best_name(최우수 모델명)에 해당하는 모델 객체 추출
    X_all = model_state["X"]  # 전체 feature(피처) 행렬 X_all 조회
    
    # --- 분석 대상 행 선택: 자가용 우선, 없으면 첫 번째 행 ---
    local_row = matched[matched["용도"] == "자가용"]  # "자가용" 용도 행 필터
    if local_row.empty:  # 자가용 데이터가 없으면
        local_row = matched.head(1)  # 매칭된 첫 번째 행 사용
    
    r_name = local_row.iloc[0]["지역"]  # 분석 대상 행정구역명
    usage = local_row.iloc[0]["용도"]  # 분석 대상 용도
    
    feature_cols = model_state["feature_columns"]  # 모델 feature(피처) 컬럼명 리스트
    local_x = X_all[(final_data["지역"] == r_name) & (final_data["용도"] == usage)].head(1)  # 해당 지역·용도의 feature(피처) 1행 추출
    if local_x.empty:  # feature(피처) 데이터가 없으면
        return f"'{r_name} ({usage})' 지역에 대한 피처 데이터를 찾지 못해 SHAP 분석을 수행할 수 없습니다."  # 에러 메시지 반환
        
    sample = X_all.sample(min(60, len(X_all)), random_state=42)  # background(배경) 샘플 60건 추출 (SHAP 기준점용)
    
    try:  # SHAP 계산 try 블록 시작
        base_val, vals = get_cached_local_shap(model_state["best_name"], best_model, sample, local_x)  # local SHAP(로컬 설명) 값 계산
        
        contributions = []  # feature(피처)별 기여도 리스트 초기화
        for col, val in zip(feature_cols, vals):  # feature(피처)명과 SHAP 값 zip(병렬 순회)
            contributions.append({  # 기여도 dict(딕셔너리) 추가
                "Feature": col,  # feature(피처) 컬럼명
                "SHAP_Value": float(val)  # SHAP(설명 가능 AI) 기여도 값
            })
            
        df_contrib = pd.DataFrame(contributions)  # contributions(기여도) 리스트를 DataFrame(데이터프레임)으로 변환
        df_contrib["Abs_SHAP"] = df_contrib["SHAP_Value"].abs()  # 절대값 컬럼 추가 (정렬용)
        df_contrib = df_contrib.sort_values("Abs_SHAP", ascending=False)  # Abs_SHAP(절대 기여도) 내림차순 정렬
        
        lines = []  # 마크다운 결과 줄 리스트 초기화
        lines.append(f"### 🧠 {r_name} ({usage}) 부하 예측 SHAP 기여도 분석")  # 제목 줄 추가
        lines.append(f"- **기본 기대 예측값(Base Value)**: {base_val:.2f}")  # base value(기준 예측값) 줄 추가
        lines.append(f"- **최종 모델 예측 부하지수**: {local_row.iloc[0]['전력_부하지수']:.2f}\n")  # 최종 예측값 줄 추가
        lines.append("#### 📊 주요 피처별 영향력 분석 (Top 5)")  # Top 5 소제목 줄 추가
        
        for _, r in df_contrib.head(5).iterrows():  # 상위 5개 feature(피처) 순회
            f_name = r["Feature"]  # feature(피처)명
            s_val = r["SHAP_Value"]  # SHAP(설명 가능 AI) 기여도 값
            direction = "🔺 증가 기여" if s_val >= 0 else "🔻 감소 기여"  # direction(기여 방향) 텍스트 결정
            lines.append(f"  - **{f_name}**: {s_val:+.4f} ({direction})")  # feature(피처)별 기여도 줄 추가
            
        # --- SHAP 차트 렌더링용 데이터를 session_state(세션 상태)에 저장 ---
        st.session_state["last_shap_chart"] = {  # last_shap_chart(최근 SHAP 차트) dict(딕셔너리) 저장
            "region": r_name,  # 행정구역명
            "usage": usage,  # 용도
            "features": df_contrib["Feature"].tolist(),  # feature(피처)명 리스트
            "values": df_contrib["SHAP_Value"].tolist()  # SHAP 값 리스트
        }
        
        gc.collect()  # gc.collect(가비지 컬렉션) 강제 실행
        return "\n".join(lines)  # 마크다운 결과 문자열 반환
        
    except Exception as e:  # SHAP 계산 중 예외 포착
        return f"SHAP 계산 중 에러가 발생했습니다: {e}"  # 에러 메시지 반환

# --- 수요반응(DR) 시뮬레이션 Function Calling(함수 호출) 대상 함수 ---
@st.cache_data(show_spinner=False, ttl=600)  # cache_data(데이터 캐시) 데코레이터 — 600초 TTL(유효시간) 캐싱
def run_dr_simulation(region: str, intervention_type: str = "V2G_Peak_Shaving") -> str:  # DR(수요반응) 정책 시뮬레이션 결과 반환
    """
    지정된 수도권 행정구역에 스마트 그리드 수요반응(DR) 또는 피크 컷 정책을 도입했을 때의 
    예측 전력 부하지수 완화율 및 과부하 지연 효과를 시뮬레이션합니다.
    
    Args:
        region: 분석할 행정구역명 (예: '안양시 동안구', '안양시' 등)
        intervention_type: 적용할 정책 개입 타입
                           ('V2G_Peak_Shaving' 또는 'Smart_Charging_50')
    Returns:
        수요반응 시뮬레이션 결과에 대한 상세 요약 보고서
    """
    import numpy as np  # numpy(넘파이) 수치 연산 라이브러리 import(임포트)
    import pandas as pd  # pandas(판다스) DataFrame(데이터프레임) 처리 라이브러리 import(임포트)
    import gc  # gc(가비지 컬렉션) 모듈 import(임포트)
    from utils.optimization import calculate_single_region_trajectory  # 단일 지역 trajectory(궤적) 계산 함수 import(임포트)

    final_data = st.session_state.get("final_data")  # session_state(세션 상태)에서 최종 분석 데이터 조회
    if final_data is None:  # 데이터가 없으면
        return "시스템 에러: 데이터가 세션에 존재하지 않습니다."  # 에러 메시지 반환

    # --- 지역명 매칭: "지역" → "시군구" 순으로 fuzzy match(퍼지 매칭) ---
    matched = final_data[final_data["지역"].str.contains(region, case=False, na=False)]  # "지역" 컬럼 부분 일치 검색
    if matched.empty:  # "지역" 컬럼 매칭 실패 시
        matched = final_data[final_data["시군구"].str.contains(region, case=False, na=False)]  # "시군구" 컬럼으로 재검색
        if matched.empty:  # 두 컬럼 모두 매칭 실패 시
            return f"검색된 지역 '{region}'을 찾을 수 없습니다. 경기 안양시 동안구 등 수도권 내 정확한 행정구역명을 입력해 주세요."  # 안내 메시지 반환

    # --- 정책 타입별 부하 감축 multiplier(배율) 결정 ---
    if "V2G" in intervention_type:  # V2G(차량-그리드) 정책인 경우
        reduction_multiplier = 0.70  # 피크 부하 30% 감축 (70% 잔존)
        policy_name = "V2G 양방향 충방전 Peak Shaving (피크 부하 30% 감축)"  # 정책명 텍스트
    else:  # Smart Charging(스마트 충전) 정책인 경우
        reduction_multiplier = 0.80  # 피크 부하 20% 감축 (80% 잔존)
        policy_name = "Smart Charging 50% 분배제한 (피크 부하 20% 감축)"  # 정책명 텍스트

    critical_threshold = float(final_data["전력_부하지수"].quantile(0.8))  # quantile(분위수) 80% = 고위험 임계치

    lines = []  # 마크다운 결과 줄 리스트 초기화
    lines.append(f"### 🔋 {region} 스마트 그리드 수요반응(DR) 시뮬레이션 결과")  # 제목 줄 추가
    lines.append(f"- **적용 정책**: {policy_name}")  # 정책 요약 줄 추가
    lines.append(f"- **전력 부하 임계치**: {critical_threshold:,.2f} (상위 20% 고위험 기준선)\n")  # 임계치 줄 추가

    # --- 매칭된 각 세부 구군별 DR(수요반응) 시뮬레이션 루프 ---
    for idx, row in matched.iterrows():  # matched(매칭된) DataFrame(데이터프레임) 행 순회
        r_name = row["지역"]  # 행정구역명 추출
        usage = row["용도"]  # 용도 추출
        before_load = float(row["전력_부하지수"])  # 정책 적용 전 전력 부하지수
        base_load = float(row["총_전력판매량"])  # 총 전력판매량(기준 부하)
        capacity = float(row["총용량_kW"])  # 현재 충전기 총용량(kW)

        sim_load = base_load * reduction_multiplier  # sim_load(시뮬레이션 부하) = 기준 부하 × 감축 배율
        after_load = sim_load / capacity if capacity > 0 else 0  # 정책 적용 후 부하지수 = sim_load / capacity
        reduction_pct = (before_load - after_load) / before_load * 100 if before_load > 0 else 0  # reduction_pct(감소율 %) 계산

        # --- 정책 적용 후 trajectory(궤적) 계산 ---
        traj_df, overload_before, overload_after = calculate_single_region_trajectory(  # 정책 적용 후 과부하 시점
            sim_load, capacity, 0.05, 0.0, critical_threshold  # sim_load, capacity, growth, added_kw=0, threshold
        )
        _, overload_before_orig, _ = calculate_single_region_trajectory(  # 정책 적용 전 원래 과부하 시점
            base_load, capacity, 0.05, 0.0, critical_threshold  # base_load, capacity, growth, added_kw=0, threshold
        )

        delay_years = overload_after - overload_before_orig  # delay_years(과부하 지연 연수)
        delay_text = f"+{delay_years}년 지연" if delay_years > 0 else "변동 없음"  # 지연 효과 텍스트
        before_txt = f"{overload_before_orig}년 뒤" if overload_before_orig < 15 else "안전 (15년+)"  # 정책 전 과부하 시점
        after_txt = f"{overload_after}년 뒤" if overload_after < 15 else "안전 (15년+)"  # 정책 후 과부하 시점

        lines.append(f"#### 📍 {r_name} ({usage} 기준)")  # 구군별 소제목 줄 추가
        lines.append(f"  - **전력 부하지수**: {before_load:,.2f} ➡️ **{after_load:,.2f}** ({reduction_pct:+.1f}% 감소)")  # 부하지수 전/후 비교
        lines.append(f"  - **과부하 도달 시점**: {before_txt} ➡️ **{after_txt}** (지연 효과: {delay_text})")  # 과부하 시점 비교
        lines.append("")  # 구분용 빈 줄 추가

    gc.collect()  # gc.collect(가비지 컬렉션) 강제 실행
    return "\n".join(lines)  # 마크다운 결과 문자열 반환

# --- 상위 N개 고위험 지역 조회 Function Calling(함수 호출) 대상 함수 ---
@st.cache_data(show_spinner=False, ttl=600)  # cache_data(데이터 캐시) 데코레이터 — 600초 TTL(유효시간) 캐싱
def get_top_regions(n: int = 10, metric: str = "전력_부하지수") -> str:  # metric(지표) 기준 상위 N개 지역 통계 반환
    """
    수도권 내에서 전력 부하지수 또는 인프라 부하지수가 가장 높은 상위 N개 지역의 상세 통계 테이블을 조회합니다.
    
    Args:
        n: 조회할 상위 지역 개수 (예: 10)
        metric: 정렬 기준이 되는 지표 ('전력_부하지수' 또는 '인프라_부하지수')
    Returns:
        상위 N개 지역의 통계 정보가 담긴 마크다운 표
    """
    import pandas as pd  # pandas(판다스) DataFrame(데이터프레임) 처리 라이브러리 import(임포트)
    import gc  # gc(가비지 컬렉션) 모듈 import(임포트)

    final_data = st.session_state.get("final_data")  # session_state(세션 상태)에서 최종 분석 데이터 조회
    if final_data is None:  # 데이터가 없으면
        return "시스템 에러: 데이터가 세션에 존재하지 않습니다."  # 에러 메시지 반환

    # --- 지표명 매핑: 영문/국문 키워드 허용 ---
    if "인프라" in metric or "infra" in metric.lower():  # metric(지표)에 "인프라" 또는 "infra" 포함 시
        sort_col = "인프라_부하지수"  # 인프라 부하지수 컬럼으로 정렬
    else:  # 그 외의 경우
        sort_col = "전력_부하지수"  # 전력 부하지수 컬럼으로 정렬 (기본값)

    if sort_col not in final_data.columns:  # sort_col(정렬 컬럼)이 DataFrame(데이터프레임)에 없으면
        return f"에러: 지표 '{sort_col}'를 데이터셋에서 찾을 수 없습니다."  # 에러 메시지 반환

    # --- 상위 N개 지역 정렬 및 추출 ---
    top_df = final_data.sort_values(sort_col, ascending=False).head(n)  # sort_col(정렬 컬럼) 내림차순 상위 n건
    
    lines = []  # 마크다운 결과 줄 리스트 초기화
    lines.append(f"### 📊 수도권 {sort_col} 상위 {n}개 우려지역")  # 제목 줄 추가
    lines.append("| 순위 | 지역명 | 용도 | 전력 부하지수 | 인프라 부하지수 | 전체 충전기 대수 | 총용량 (kW) |")  # 마크다운 테이블 헤더
    lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: |")  # 마크다운 테이블 구분선
    
    for idx, (_, row) in enumerate(top_df.iterrows(), 1):  # top_df(상위 지역) 행 순회 (1부터 순위 부여)
        r_name = row["지역"]  # 행정구역명
        usage = row["용도"]  # 용도
        p_load = float(row["전력_부하지수"])  # 전력 부하지수
        i_load = float(row["인프라_부하지수"])  # 인프라 부하지수
        chargers = int(row["전체_충전기대수"])  # 전체 충전기 대수
        capacity = float(row["총용량_kW"])  # 총용량(kW)
        lines.append(f"| {idx} | {r_name} | {usage} | {p_load:,.2f} | {i_load:,.2f} | {chargers:,}대 | {capacity:,.1f} kW |")  # 테이블 데이터 행 추가

    gc.collect()  # gc.collect(가비지 컬렉션) 강제 실행
    return "\n".join(lines)  # 마크다운 테이블 문자열 반환

# --- PDF 보고서 생성 Function Calling(함수 호출) 대상 함수 ---
def generate_report_from_conversation(region: str) -> str:  # 대화 중 요청된 지역의 PDF(포터블 문서) 보고서 생성
    """
    사용자가 특정 행정구역 분석 보고서 출력을 원할 때 호출되어, 해당 지역의 상세 통계, 예측 부하지수 및 정책 제안이 포함된 맞춤형 PDF 보고서를 생성하고 다운로드 링크를 준비합니다.
    
    Args:
        region: 보고서를 발간할 행정구역명 (예: '안양시', '부천시' 등)
    Returns:
        보고서 PDF 파일 생성 완료 메시지 및 안내 문구
    """
    from utils.pdf_generator import generate_regional_report_pdf  # 지역별 PDF(포터블 문서) 보고서 생성 함수 import(임포트)
    
    final_data = st.session_state.get("final_data")  # session_state(세션 상태)에서 최종 분석 데이터 조회
    hourly_data = st.session_state.get("hourly_data")  # session_state(세션 상태)에서 시간대별 데이터 조회
    if final_data is None:  # 데이터가 없으면
        return "시스템 에러: 데이터를 로드하지 못했습니다."  # 에러 메시지 반환
        
    try:  # PDF 생성 try 블록 시작
        pdf_bytes = generate_regional_report_pdf(region, final_data, hourly_data)  # PDF(포터블 문서) 바이트 생성
        st.session_state["pdf_report_bytes"] = pdf_bytes  # pdf_report_bytes(보고서 바이트) session_state(세션 상태) 저장
        st.session_state["pdf_report_region"] = region  # pdf_report_region(보고서 지역명) 저장
        st.session_state["pending_pdf_bytes"] = pdf_bytes  # pending_pdf_bytes(대기 중 PDF) — 채팅 메시지 첨부용
        st.session_state["pending_pdf_region"] = region  # pending_pdf_region(대기 중 PDF 지역명) 저장
        
        # --- session_state(세션 상태)의 generated_reports(생성된 보고서) dict(딕셔너리)에 누적 저장 ---
        if "generated_reports" not in st.session_state:  # generated_reports(생성된 보고서) 키가 없으면
            st.session_state["generated_reports"] = {}  # 빈 dict(딕셔너리) 초기화
        st.session_state["generated_reports"][region] = pdf_bytes  # 지역명을 key(키)로 PDF 바이트 저장
        
        return f"성공적으로 {region} 지역의 맞춤형 관제 분석 보고서 PDF(3페이지 분량)가 생성되었습니다. 아래 다운로드 버튼을 확인하십시오."  # 성공 메시지 반환
    except Exception as e:  # PDF 생성 중 예외 포착
        return f"보고서 PDF 생성 실패: {str(e)}"  # 에러 메시지 반환

# --- Groq(그록) OpenAI 호환 API(애플리케이션 프로그래밍 인터페이스) 호출 함수 ---
def call_openai_compatible_api(provider: str, api_key: str, messages: list, tools: list = None) -> tuple:  # Groq API 호출 후 (content, tool_calls) tuple(튜플) 반환
    """
    Groq(그록) OpenAI-compatible(호환) chat completions(채팅 완성) API(애플리케이션 프로그래밍 인터페이스)를 호출합니다.
    Returns:
        (assistant_message_content, tool_calls_list) — (assistant(어시스턴트) text(텍스트), tool_calls(함수 호출) list(리스트))
    """
    import requests  # requests(HTTP 요청) 라이브러리 import(임포트)
    import json  # json(JSON) 직렬화 모듈 import(임포트)
    
    url = "https://api.groq.com/openai/v1/chat/completions"  # Groq(그록) OpenAI 호환 chat completions(채팅 완성) endpoint(엔드포인트)
    model = "llama-3.3-70b-versatile"  # 사용할 Llama(라마) 3.3 70B 모델명

    headers = {  # HTTP request(요청) 헤더 dict(딕셔너리)
        "Authorization": f"Bearer {api_key}",  # Bearer(베어러) 토큰 인증
        "Content-Type": "application/json"  # JSON(자바스크립트 객체 표기법) content type(콘텐츠 타입)
    }
    
    payload = {  # API request(요청) body(본문) dict(딕셔너리)
        "model": model,  # LLM(대규모 언어 모델) 모델명
        "messages": messages,  # 대화 messages(메시지) 리스트
        "temperature": 0.3  # temperature(온도) — 응답 다양성 제어 (낮을수록 결정적)
    }
    
    if tools:  # tools(함수 호출 도구) 리스트가 제공된 경우
        payload["tools"] = tools  # tools(함수 호출 도구) 추가
        payload["tool_choice"] = "auto"  # tool_choice(도구 선택) — 모델이 자동으로 함수 호출 여부 결정
        
    res = requests.post(url, headers=headers, json=payload, timeout=30)  # POST(포스트) HTTP 요청 전송 (30초 timeout(타임아웃))
    res.raise_for_status()  # HTTP status(상태) 코드가 4xx/5xx이면 예외 발생
    res_json = res.json()  # response(응답) JSON(자바스크립트 객체 표기법) 파싱
    
    choice = res_json["choices"][0]  # 첫 번째 completion(완성) choice(선택) 추출
    message = choice["message"]  # assistant(어시스턴트) message(메시지) 추출
    content = message.get("content") or ""  # text content(텍스트 콘텐츠) 추출 (없으면 빈 문자열)
    tool_calls = message.get("tool_calls") or []  # tool_calls(함수 호출) 리스트 추출 (없으면 빈 리스트)
    
    return content, tool_calls  # (content, tool_calls) tuple(튜플) 반환

# --- Gemini(제미니) API 클라이언트 초기화 함수 ---
def get_gemini_client():  # Gemini(제미니) GenerativeModel(생성 모델) 객체 반환 (실패 시 None)
    api_key = st.secrets.get("GEMINI_API_KEY")  # st.secrets(시크릿)에서 GEMINI_API_KEY 조회 (1순위)
    
    if not api_key:  # secrets(시크릿)에 키가 없으면
        api_key = st.session_state.get("user_gemini_api_key")  # session_state(세션 상태)의 사용자 입력 키 fallback(대체)
        
    if not api_key:  # API key(키)가 여전히 없으면
        return None  # None 반환
        
    try:  # Gemini(제미니) 초기화 try 블록 시작
        genai.configure(api_key=api_key)  # genai(제미니 SDK) API key(키) 설정
        
        # --- API key(키)로 사용 가능한 model(모델) 목록 programmatic(프로그래밍) 탐색 ---
        available_model_names = []  # 사용 가능한 model(모델)명 리스트 초기화
        try:  # model(모델) 목록 조회 try 블록
            for m in genai.list_models():  # genai.list_models(모델 목록) 순회
                if hasattr(m, 'supported_generation_methods') and "generateContent" in m.supported_generation_methods:  # generateContent(콘텐츠 생성) 지원 model(모델)만
                    available_model_names.append(m.name)  # model(모델)명 리스트에 추가
        except Exception:  # model(모델) 목록 조회 실패 시
            pass  # 무시하고 fallback(대체) model(모델) 사용
            
        candidate_models = [  # 우선순위 candidate(후보) model(모델) 리스트
            "gemini-1.5-flash",  # Gemini 1.5 Flash(플래시) — 빠른 응답
            "gemini-1.5-flash-latest",  # Gemini 1.5 Flash 최신 버전
            "gemini-1.5-pro",  # Gemini 1.5 Pro(프로) — 고성능
            "gemini-1.0-pro",  # Gemini 1.0 Pro(프로) — 구버전 fallback(대체)
        ]
        
        selected_model_name = None  # 선택된 model(모델)명 초기화
        for candidate in candidate_models:  # candidate(후보) model(모델) 순회
            full_candidate = candidate if candidate.startswith("models/") else f"models/{candidate}"  # "models/" prefix(접두사) 보장
            if full_candidate in available_model_names:  # 사용 가능한 model(모델) 목록에 존재하면
                selected_model_name = candidate  # 해당 model(모델) 선택
                break  # loop(루프) 종료
                
        if not selected_model_name:  # candidate(후보) 중 매칭된 model(모델)이 없으면
            if available_model_names:  # 사용 가능한 model(모델) 목록이 있으면
                selected_model_name = available_model_names[0].replace("models/", "")  # 첫 번째 model(모델) 사용
            else:  # model(모델) 목록 조회도 실패했으면
                selected_model_name = "gemini-1.5-flash"  # 기본 fallback(대체) model(모델)
                
        st.session_state["gemini_selected_model"] = selected_model_name  # 선택된 model(모델)명 session_state(세션 상태) 저장
        model = genai.GenerativeModel(selected_model_name)  # GenerativeModel(생성 모델) 객체 생성
        return model  # model(모델) 객체 반환
    except Exception as e:  # Gemini(제미니) 초기화 중 예외 포착
        st.error(f"Gemini API 설정 중 오류가 발생했습니다: {e}")  # error(에러) 메시지 UI 표시
        return None  # None 반환

# --- 대시보드 상태를 LLM(대규모 언어 모델) system prompt(시스템 프롬프트)용 JSON(자바스크립트 객체 표기법) context(컨텍스트)로 변환 ---
def build_system_context(filtered_data, model_state, control_mode, hw_data=None):  # 현재 대시보드 상태 요약 JSON(자바스크립트 객체 표기법) 문자열 반환
    """
    대시보드 현재 상태를 JSON(자바스크립트 객체 표기법) 형식 context(컨텍스트) 텍스트로 변환하여
    Gemini(제미니) system instruction(시스템 지침)에 주입합니다.
    """
    context = {  # LLM(대규모 언어 모델)에 주입할 context(컨텍스트) dict(딕셔너리) 초기화
        "dashboard_mode": control_mode,  # 현재 dashboard(대시보드) control_mode(관제 모드)
    }
    
    if control_mode == "도심 행정구역 관제" and filtered_data is not None:  # 도심 관제 모드이고 filtered_data(필터 데이터)가 있으면
        # --- 도심(urban) context(컨텍스트) 집계 ---
        total_districts = len(filtered_data)  # 필터링된 행정구역 총 수
        avg_power_load = float(filtered_data["전력_부하지수"].mean()) if "전력_부하지수" in filtered_data.columns else 0.0  # 평균 전력 부하지수
        avg_infra_load = float(filtered_data["인프라_부하지수"].mean()) if "인프라_부하지수" in filtered_data.columns else 0.0  # 평균 인프라 부하지수
        
        # --- TOP 3 고위험 지역 식별 ---
        top3 = []  # TOP 3 지역 리스트 초기화
        if "전력_부하지수" in filtered_data.columns:  # "전력_부하지수" 컬럼이 존재하면
            top3_df = filtered_data.sort_values("전력_부하지수", ascending=False).head(3)  # 전력 부하지수 상위 3건
            for _, row in top3_df.iterrows():  # TOP 3 행 순회
                top3.append({  # 지역 정보 dict(딕셔너리) 추가
                    "지역": row.get("지역", f"{row.get('시도', '')} {row.get('구군', '')}"),  # 행정구역명
                    "용도": row.get("용도", "N/A"),  # 용도
                    "전력_부하지수": float(row.get("전력_부하지수", 0.0)),  # 전력 부하지수
                    "인프라_부하지수": float(row.get("인프라_부하지수", 0.0))  # 인프라 부하지수
                })
        
        # --- ML(머신러닝) model(모델) 평가 정보 추출 ---
        best_model_name = "N/A"  # best model(최우수 모델)명 기본값
        test_rmse = 0.0  # Test RMSE(테스트 평균제곱근오차) 기본값
        if model_state and "best_name" in model_state:  # model_state(모델 상태)에 best_name(최우수 모델명)이 있으면
            best_model_name = model_state["best_name"]  # best model(최우수 모델)명 추출
            metrics_df = model_state.get("metrics")  # metrics(평가 지표) DataFrame(데이터프레임) 조회
            if metrics_df is not None and not metrics_df.empty:  # metrics(평가 지표)가 존재하면
                test_rmse_row = metrics_df[(metrics_df["Model"] == best_model_name) & (metrics_df["Split"] == "Test")]  # Test split(분할) RMSE 행 필터
                if not test_rmse_row.empty:  # Test RMSE 행이 있으면
                    test_rmse = float(test_rmse_row["RMSE"].values[0])  # RMSE(평균제곱근오차) 값 추출
            
        context["urban_summary"] = {  # urban_summary(도심 요약) dict(딕셔너리) 추가
            "총_필터링된_행정구역_수": total_districts,  # 필터링된 행정구역 수
            "평균_전력_부하지수": round(avg_power_load, 4),  # 평균 전력 부하지수
            "평균_인프라_부하지수": round(avg_infra_load, 4),  # 평균 인프라 부하지수
            "TOP_3_고위험_우려지역": top3,  # TOP 3 고위험 지역 리스트
            "최적_예측_모델": best_model_name,  # 최우수 예측 model(모델)명
            "예측_오차_RMSE": round(test_rmse, 4)  # Test RMSE(테스트 평균제곱근오차)
        }
        
    elif control_mode == "고속도로망 최적화" and hw_data is not None:  # 고속도로 관제 모드이고 hw_data(고속도로 데이터)가 있으면
        # --- 고속도로(highway) context(컨텍스트) 집계 ---
        total_nodes = len(hw_data)  # 고속도로 node(노드) 총 수
        avg_traffic = float(hw_data["교통량"].mean()) if "교통량" in hw_data.columns else 0.0  # 평균 교통량
        context["highway_summary"] = {  # highway_summary(고속도로 요약) dict(딕셔너리) 추가
            "총_고속도로_노드_수": total_nodes,  # 총 node(노드) 수
            "평균_교통량": round(avg_traffic, 2)  # 평균 교통량
        }
        
    return json.dumps(context, ensure_ascii=False, indent=2)  # context(컨텍스트) dict(딕셔너리)를 JSON(자바스크립트 객체 표기법) 문자열로 직렬화하여 반환

# --- AI 관제비서 메인 UI(사용자 인터페이스) 렌더링 함수 ---
def render_ai_assistant(filtered_data, model_state, control_mode, hw_data=None):  # AI 관제비서 채팅 UI(사용자 인터페이스) 렌더링
    # --- AI 관제비서 헤더 HTML/CSS(하이퍼텍스트/캐스케이딩 스타일시트) 블록 시작 ---
    st.markdown("""
        <style>
        .ai-header {
            background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
            padding: 25px;
            border-radius: 16px;
            color: white;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2);
        }
        .ai-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 8px;
            font-family: 'Inter', sans-serif;
        }
        .ai-subtitle {
            font-size: 1.05rem;
            opacity: 0.9;
        }
        .card-container {
            display: flex;
            gap: 15px;
            margin-bottom: 25px;
        }
        .info-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 15px;
            flex: 1;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }
        .info-card h4 {
            margin-top: 0;
            color: #1E293B;
            font-size: 1rem;
            border-bottom: 2px solid #3B82F6;
            padding-bottom: 6px;
            margin-bottom: 10px;
        }
        .info-card p {
            font-size: 0.9rem;
            color: #475569;
            margin: 0;
            line-height: 1.4;
        }
        </style>
        <div class="ai-header">
            <div class="ai-title">🤖 AI 관제 어시스턴트 (EV-Charge AI)</div>
            <div class="ai-subtitle">Google Gemini 1.5 Flash 모델 기반의 대화형 충전소 입지 및 부하 데이터 분석 비서입니다.</div>
        </div>
    """, unsafe_allow_html=True)  # HTML(하이퍼텍스트) 블록 렌더링 (unsafe_allow_html=True: HTML 태그 허용)
    # --- AI 관제비서 헤더 HTML/CSS(하이퍼텍스트/캐스케이딩 스타일시트) 블록 끝 ---

    # --- API(애플리케이션 프로그래밍 인터페이스) provider(제공자) 및 key(키) 조회 ---
    current_provider = st.session_state.get("ai_provider", "Gemini (Google)")  # session_state(세션 상태)에서 현재 AI provider(제공자) 조회
    if current_provider not in ["Gemini (Google)", "Groq (Llama 3)"]:  # 유효하지 않은 provider(제공자) 값이면
        current_provider = "Gemini (Google)"  # 기본값 Gemini(제미니)로 reset(리셋)
        st.session_state["ai_provider"] = current_provider  # session_state(세션 상태)에 저장
    
    # --- 1. st.secrets(시크릿)에서 API key(키) 조회 (backend(백엔드) 전용, frontend(프론트엔드) 미노출) ---
    gemini_secret_key = st.secrets.get("GEMINI_API_KEY")  # Gemini(제미니) secret(시크릿) key(키)
    groq_secret_key = st.secrets.get("GROQ_API_KEY")  # Groq(그록) secret(시크릿) key(키)
    
    # --- 현재 provider(제공자)에 대한 active key(활성 키) 결정 ---
    active_key = None  # active_key(활성 API 키) 초기화
    if current_provider == "Gemini (Google)":  # Gemini(제미니) provider(제공자)인 경우
        if gemini_secret_key:  # secret(시크릿) key(키)가 있으면
            active_key = gemini_secret_key  # secret(시크릿) key(키) 사용
        else:  # secret(시크릿) key(키)가 없으면
            active_key = st.session_state.get("user_gemini_api_key")  # 사용자 입력 key(키) fallback(대체)
    else:  # Groq (Llama 3) provider(제공자)인 경우
        if groq_secret_key:  # secret(시크릿) key(키)가 있으면
            active_key = groq_secret_key  # secret(시크릿) key(키) 사용
        else:  # secret(시크릿) key(키)가 없으면
            active_key = st.session_state.get("user_groq_api_key")  # 사용자 입력 key(키) fallback(대체)

    # --- API 및 engine(엔진) 설정 expander(확장 패널) UI ---
    with st.expander("⚙️ AI 관제비서 API 및 엔진 설정", expanded=not active_key):  # active_key(활성 키) 없으면 expander(확장 패널) 자동 펼침
        provider_col, key_col = st.columns([1, 2])  # provider(제공자) / key(키) 입력 2-column(열) 레이아웃
        with provider_col:  # provider(제공자) 선택 column(열)
            selected_prov = st.selectbox(  # selectbox(선택상자) — AI provider(제공자) 선택 UI
                "AI 서비스 제공자",
                ["Gemini (Google)", "Groq (Llama 3)"],
                index=["Gemini (Google)", "Groq (Llama 3)"].index(current_provider),
                key="ai_provider_input"
            )
            if selected_prov != current_provider:  # provider(제공자) 변경 시
                st.session_state["ai_provider"] = selected_prov  # session_state(세션 상태) 업데이트
                st.rerun()  # rerun(재실행) — provider(제공자) 전환 반영
                
        with key_col:  # API key(키) 입력 column(열)
            has_secret = False  # secret(시크릿) key(키) 존재 여부 flag(플래그)
            if selected_prov == "Gemini (Google)" and gemini_secret_key:  # Gemini(제미니) secret(시크릿) key(키) 존재
                has_secret = True  # has_secret(시크릿 존재) flag(플래그) 설정
            elif selected_prov == "Groq (Llama 3)" and groq_secret_key:  # Groq(그록) secret(시크릿) key(키) 존재
                has_secret = True  # has_secret(시크릿 존재) flag(플래그) 설정
                
            if has_secret:  # secret(시크릿) key(키)가 등록되어 있으면
                st.success("✅ **API Key가 시스템 설정(Secrets)에 안전하게 등록되어 있습니다.**")  # success(성공) 메시지 표시
                st.caption("보안을 위해 실제 API 키는 브라우저에 표시되거나 전송되지 않습니다.")  # 보안 안내 caption(캡션)
            else:  # secret(시크릿) key(키)가 없으면 사용자 입력 UI 표시
                if selected_prov == "Gemini (Google)":  # Gemini(제미니) provider(제공자)인 경우
                    gemini_key = st.text_input(  # text_input(텍스트 입력) — Gemini API key(키) 입력 UI
                        "Gemini API Key 입력 (임시)",
                        type="password",  # password(비밀번호) 타입 — 입력값 마스킹
                        placeholder="API Key를 입력하세요",
                        value=st.session_state.get("user_gemini_api_key", ""),
                        key="user_gemini_key_input_new"
                    )
                    if gemini_key and gemini_key != st.session_state.get("user_gemini_api_key"):  # key(키) 변경 감지
                        st.session_state["user_gemini_api_key"] = gemini_key  # session_state(세션 상태)에 key(키) 저장
                        st.rerun()  # rerun(재실행) — key(키) 반영
                else:  # Groq (Llama 3) provider(제공자)인 경우
                    groq_key = st.text_input(  # text_input(텍스트 입력) — Groq API key(키) 입력 UI
                        "Groq API Key 입력 (임시)",
                        type="password",  # password(비밀번호) 타입 — 입력값 마스킹
                        placeholder="API Key를 입력하세요",
                        value=st.session_state.get("user_groq_api_key", ""),
                        key="user_groq_key_input_new"
                    )
                    if groq_key and groq_key != st.session_state.get("user_groq_api_key"):  # key(키) 변경 감지
                        st.session_state["user_groq_api_key"] = groq_key  # session_state(세션 상태)에 key(키) 저장
                        st.rerun()  # rerun(재실행) — key(키) 반영
                    
    if not active_key:  # active_key(활성 API 키)가 없으면
        st.warning(f"⚠️ **{current_provider} API 키가 등록되지 않았습니다.**")  # warning(경고) 메시지 표시
        st.info("시작하려면 위의 'AI 관제비서 API 및 엔진 설정' 창을 열고 API 키를 입력해 주세요.")  # info(안내) 메시지 표시
        return  # key(키) 없이는 채팅 UI 렌더링 중단
        
    # --- Gemini(제미니) provider(제공자) 선택 시 model(모델) 설정 ---
    if current_provider == "Gemini (Google)":  # Gemini(제미니) provider(제공자)인 경우
        try:  # Gemini(제미니) model(모델) 설정 try 블록
            genai.configure(api_key=active_key)  # genai(제미니 SDK) API key(키) 설정
            # --- API key(키)로 사용 가능한 model(모델) 목록 programmatic(프로그래밍) 탐색 ---
            available_model_names = []  # 사용 가능한 model(모델)명 리스트 초기화
            try:  # model(모델) 목록 조회 try 블록
                for m in genai.list_models():  # genai.list_models(모델 목록) 순회
                    if hasattr(m, 'supported_generation_methods') and "generateContent" in m.supported_generation_methods:  # generateContent(콘텐츠 생성) 지원 model(모델)만
                        available_model_names.append(m.name)  # model(모델)명 리스트에 추가
            except Exception:  # model(모델) 목록 조회 실패 시
                pass  # 무시하고 fallback(대체) model(모델) 사용
                
            candidate_models = [  # 우선순위 candidate(후보) model(모델) 리스트
                "gemini-1.5-flash",
                "gemini-1.5-flash-latest",
                "gemini-1.5-pro",
                "gemini-1.0-pro",
            ]
            
            selected_model_name = None  # 선택된 model(모델)명 초기화
            for candidate in candidate_models:  # candidate(후보) model(모델) 순회
                full_candidate = candidate if candidate.startswith("models/") else f"models/{candidate}"  # "models/" prefix(접두사) 보장
                if full_candidate in available_model_names:  # 사용 가능한 model(모델) 목록에 존재하면
                    selected_model_name = candidate  # 해당 model(모델) 선택
                    break  # loop(루프) 종료
                    
            if not selected_model_name:  # candidate(후보) 중 매칭된 model(모델)이 없으면
                if available_model_names:  # 사용 가능한 model(모델) 목록이 있으면
                    selected_model_name = available_model_names[0].replace("models/", "")  # 첫 번째 model(모델) 사용
                else:  # model(모델) 목록 조회도 실패했으면
                    selected_model_name = "gemini-1.5-flash"  # 기본 fallback(대체) model(모델)
                    
            st.session_state["gemini_selected_model"] = selected_model_name  # 선택된 model(모델)명 session_state(세션 상태) 저장
        except Exception as e:  # Gemini(제미니) 설정 중 예외 포착
            st.error(f"Gemini API 설정 중 오류가 발생했습니다: {e}")  # error(에러) 메시지 UI 표시
            return  # 설정 실패 시 렌더링 중단

    # --- 기능 안내 info card(정보 카드) 3-column(열) 레이아웃 ---
    col1, col2, col3 = st.columns(3)  # 3-column(열) 레이아웃 생성
    with col1:  # 1번째 column(열) — 실시간 데이터 연동 안내
        st.markdown("""
            <div class="info-card">
                <h4>📊 실시간 데이터 연동</h4>
                <p>현재 필터링된 대시보드의 데이터를 AI가 실시간으로 분석 모델 지식과 융합하여 맞춤 답변합니다.</p>
            </div>
        """, unsafe_allow_html=True)  # info card(정보 카드) HTML(하이퍼텍스트) 렌더링
    with col2:  # 2번째 column(열) — 시뮬레이션 질의 안내
        st.markdown("""
            <div class="info-card">
                <h4>🔮 시뮬레이션 질의</h4>
                <p>"안양시에 급속 충전소 10대를 증설하면 부하가 얼마나 줄어들까?" 같은 시나리오 영향력을 문의해 보세요.</p>
            </div>
        """, unsafe_allow_html=True)  # info card(정보 카드) HTML(하이퍼텍스트) 렌더링
    with col3:  # 3번째 column(열) — PDF 보고서 생성 안내
        st.markdown("""
            <div class="info-card">
                <h4>📋 맞춤형 보고서 생성</h4>
                <p>"안양시 보고서 만들어줘" 처럼 특정 지역 이름을 언급하면 AI가 3페이지 분량의 PDF 보고서를 자동 발간합니다.</p>
            </div>
        """, unsafe_allow_html=True)  # info card(정보 카드) HTML(하이퍼텍스트) 렌더링

    # --- 📋 발간된 보고서 다운로드 센터 (session_state(세션 상태)에 생성된 보고서가 있을 때만 노출) ---
    if "generated_reports" in st.session_state and st.session_state["generated_reports"]:  # generated_reports(생성된 보고서) dict(딕셔너리) 존재 시
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)  # 상단 여백 HTML(하이퍼텍스트) 삽입
        st.markdown("### 📥 실시간 발간 완료된 맞춤형 보고서 목록")  # 보고서 목록 section(섹션) 제목
        report_items = list(st.session_state["generated_reports"].items())  # generated_reports(생성된 보고서) dict(딕셔너리) → list(리스트) 변환
        # --- None 또는 invalid(유효하지 않은) entry(항목) 필터, bytes(바이트) 타입 보장 ---
        valid_items = [  # valid(유효한) 보고서 item(항목) 리스트
            (rn, bytes(rb)) for rn, rb in report_items  # region_name(지역명), report_bytes(보고서 바이트) tuple(튜플)
            if rb is not None and isinstance(rb, (bytes, bytearray)) and len(rb) > 0  # bytes(바이트) 타입이고 비어있지 않은 경우만
        ]
        if valid_items:  # valid(유효한) 보고서가 있으면
            cols = st.columns(min(len(valid_items), 4))  # 최대 4-column(열) download button(다운로드 버튼) 레이아웃
            for col_idx, (region_name, r_bytes) in enumerate(valid_items):  # valid_items(유효 보고서) 순회
                with cols[col_idx % len(cols)]:  # column(열) 순환 배치
                    st.download_button(  # download_button(다운로드 버튼) — PDF(포터블 문서) 다운로드 UI
                        label=f"💾 {region_name} PDF 다운로드",
                        data=r_bytes,  # PDF(포터블 문서) 바이트 데이터
                        file_name=f"{region_name}_EV_charge_report.pdf",  # download(다운로드) 파일명
                        mime="application/pdf",  # MIME(미디어 타입) — PDF(포터블 문서)
                        key=f"dl_btn_center_{region_name}_{col_idx}",  # widget(위젯) unique key(고유 키)
                        use_container_width=True  # column(열) 전체 너비 사용
                    )
        st.markdown("---")  # section(섹션) 구분선

    # --- LLM(대규모 언어 모델) system prompt(시스템 프롬프트)용 dashboard context(컨텍스트) 주입 ---
    dashboard_context = build_system_context(filtered_data, model_state, control_mode, hw_data)  # 현재 dashboard(대시보드) 상태 JSON(자바스크립트 객체 표기법) context(컨텍스트) 생성
    
    # --- system_instruction(시스템 지침) f-string — LLM(대규모 언어 모델) 역할·규칙 정의 ---
    system_instruction = f"""
    당신은 '수도권 전기차 충전소 부하 예측 및 설치 시뮬레이션 서비스'의 전문 AI 관제 비서인 'EV-Charge AI'입니다.
    사용자의 질문에 친절하고 신뢰감 있는 정부 보고서 서기 스타일의 한국어로 대답하십시오.
    
    분석 시스템 데이터셋 기준 정보:
    - 전기차 등록대수 기준일: 2025년 4월 7일 (한국교통안전공단 데이터)
    - 충전소 위경도 위치 기준일: 2025년 5월 31일 (한국전력공사 데이터)
    - 충전기 용량 기준일: 2024년 6월 3일 (한국전력공사 데이터)
    - 연월별 급속 충전량 데이터: 2015년 ~ 2025년 8월 (환경부 데이터)
    - 시간대별 충전부하 데이터: 2024년 9월 30일 (한국전력공사 데이터)
    - 즉, 본 시스템은 2024년~2025년의 최신 공공 데이터를 융합 분석한 결과입니다. 2024년 5월 16일 같은 임의의 과거 날짜나 특정 시점만을 기준으로 보고서를 작성하지 않았음을 확실히 인지하고 설명하십시오.
    
    [CRITICAL RULE - PDF REPORT GENERATION]
    - 사용자가 "보고서", "리포트", "PDF", "다운로드" 등의 단어와 함께 특정 지역(예: "강남구", "안양시", "부천시" 등)의 보고서 생성을 요청하는 경우, 당신은 절대로 스스로 보고서가 생성되었다고 답변을 날조(Hallucination)해서는 안 됩니다.
    - 이러한 요청을 받으면 **반드시** 즉시 `generate_report_from_conversation` 툴을 호출하여 실제로 PDF 파일을 생성해야 합니다.
    - 툴 호출 결과(Function Response)를 받은 후에만 "성공적으로 보고서가 생성되었습니다"라고 사용자에게 답변하십시오. 툴을 호출하지 않은 채 생성되었다고 거짓말하지 마십시오.
    
    현재 대시보드의 실시간 통계 및 예측 상태 데이터(JSON 형식):
    {dashboard_context}
    
    중요 지침 (한글 마크다운 렌더링 깨짐 방지):
    - 한글 문장에서 볼드 기호(**) 뒤에 조사(은/는/이/가/을/를/에/의 등)가 공백 없이 바로 붙으면 일부 마크다운 렌더러에서 볼드가 적용되지 않고 깨지는 현상이 발생합니다.
    - 예: "**경기 부천시 (사업자용)**이" (깨짐 우려)
    - 해결책: 볼드 강조를 사용할 때는 조사를 강조 범위 안에 포함시키십시오. (예: "**경기 부천시 (사업자용)이**" 또는 "**경기 부천시 (사업자용)** 이"와 같이 한 칸 띄우기). 가능하면 조사를 강조 범위 안으로 포함하는 방식("**경기 부천시 (사업자용)이**")을 강력히 권장합니다.
    
    답변 지침:
    1. 사용자가 특정 지역의 상태나 전체 데이터의 특징을 묻는 경우, 위의 실시간 요약 데이터(JSON)를 바탕으로 정확한 통계 수치를 인용하여 설명하십시오.
    2. 주요 위험 지역(TOP 3 고위험 우려지역 등)의 전력 및 인프라 부하지수를 해석해 주십시오.
    3. 예측 모델에 관해 묻는다면, 현재 최적 모델명과 Test RMSE 오차를 토대로 과학적 타당성을 설명해 주십시오.
    4. 분석 내용에 표나 불릿 포인트를 적극적으로 활용하여 공공 보고서처럼 가독성 높게 답변하십시오.
    5. 사용자가 상위 여러 지역(예: TOP 5, TOP 10 등)의 부하지수 목록이나 테이블을 요청하는 경우, get_top_regions 툴을 적극적으로 호출하여 실시간 통계 표를 사용자에게 제시하십시오.
    6. 사용자가 특정 행정구역에 대한 상세 분석 보고서 발간, PDF 다운로드 또는 리포트 출력을 요청하는 경우(예: "안양시 보고서 만들어줘", "부천시 PDF 다운로드"), 반드시 'generate_report_from_conversation' 툴을 호출하여 맞춤형 PDF 보고서를 생성해 주십시오.
    """  # system_instruction(시스템 지침) f-string 끝

    # --- session_state(세션 상태) chat history(채팅 기록) 초기화 ---
    if "messages" not in st.session_state:  # messages(메시지) 키가 없으면
        st.session_state.messages = []  # 빈 chat history(채팅 기록) list(리스트) 초기화

    # --- app rerun(재실행) 시 기존 chat history(채팅 기록) message(메시지) 재렌더링 ---
    for idx, message in enumerate(st.session_state.messages):  # messages(메시지) list(리스트) 순회
        with st.chat_message(message["role"]):  # chat_message(채팅 메시지) bubble(말풍선) — role(역할: user/assistant)별 스타일
            st.markdown(message["content"])  # message(메시지) text content(텍스트 콘텐츠) 마크다운 렌더링
            _pdf_b = message.get("pdf_bytes")  # message(메시지)에 첨부된 PDF(포터블 문서) bytes(바이트) 조회
            _pdf_r = message.get("pdf_region")  # message(메시지)에 첨부된 PDF(포터블 문서) region(지역명) 조회
            if _pdf_b is not None and _pdf_r and isinstance(_pdf_b, (bytes, bytearray)) and len(_pdf_b) > 0:  # valid(유효한) PDF(포터블 문서) 첨부가 있으면
                st.download_button(  # download_button(다운로드 버튼) — 채팅 message(메시지) 내 PDF 다운로드
                    label=f"⬇️ {_pdf_r} 관제 분석 보고서 다운로드 (PDF)",
                    data=bytes(_pdf_b),  # PDF(포터블 문서) 바이트 데이터
                    file_name=f"{_pdf_r}_EV_charge_report.pdf",  # download(다운로드) 파일명
                    mime="application/pdf",  # MIME(미디어 타입) — PDF(포터블 문서)
                    key=f"dl_btn_{idx}"  # widget(위젯) unique key(고유 키)
                )
            elif message["role"] == "assistant" and "generated_reports" in st.session_state:  # PDF(포터블 문서) metadata(메타데이터) 없을 때 fallback(대체) 검색
                # --- fallback(대체): message(메시지) content(콘텐츠)에서 region(지역명) 매칭하여 generated_reports(생성된 보고서) 검색 ---
                for region, report_bytes in st.session_state["generated_reports"].items():  # generated_reports(생성된 보고서) dict(딕셔너리) 순회
                    if report_bytes is None or not isinstance(report_bytes, (bytes, bytearray)) or len(report_bytes) == 0:  # invalid(유효하지 않은) entry(항목) skip(건너뜀)
                        continue  # 다음 entry(항목)로
                    short_region = region.replace("서울", "").replace("경기", "").replace("인천", "").strip()  # short_region(약칭 지역명) — 시도 prefix(접두사) 제거
                    if (region in message["content"] or short_region in message["content"]) and ("보고서" in message["content"] or "PDF" in message["content"]):  # content(콘텐츠)에 region(지역명)·보고서 키워드 포함
                        st.download_button(  # fallback(대체) download_button(다운로드 버튼)
                            label=f"📥 {region} 관제 분석 보고서 다운로드 (PDF)",
                            data=bytes(report_bytes),  # PDF(포터블 문서) 바이트 데이터
                            file_name=f"{region}_EV_charge_report.pdf",  # download(다운로드) 파일명
                            mime="application/pdf",  # MIME(미디어 타입) — PDF(포터블 문서)
                            key=f"dl_btn_fallback_{idx}_{region}"  # widget(위젯) unique key(고유 키)
                        )
                        break  # 첫 매칭 entry(항목)만 표시

    # --- 추천 질문 suggestion chip(제안 칩) UI ---
    st.markdown("##### 💡 추천 질문 바로 하기")  # suggestion(추천 질문) section(섹션) 소제목
    suggestion_cols = st.columns(3)  # 3-column(열) suggestion button(제안 버튼) 레이아웃
    
    suggestions = [  # preset(사전 설정) 추천 질문 list(리스트)
        "현재 수도권 내에서 가장 위험한 TOP 3 지역의 부하 지수를 비교해줘.",
        "학습된 예측 모델 중 최우수 성능 모델과 신뢰성에 대해 설명해줘.",
        "충전 부하가 가장 높은 병목 구간에 대한 단기 대책과 제안은 무엇인가요?"
    ]
    
    clicked_prompt = None  # clicked_prompt(클릭된 추천 질문) 초기화
    for i, suggestion in enumerate(suggestions):  # suggestions(추천 질문) list(리스트) 순회
        with suggestion_cols[i]:  # i번째 column(열)
            if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):  # suggestion button(제안 버튼) 클릭 시
                clicked_prompt = suggestion  # clicked_prompt(클릭된 추천 질문) 저장

    # --- user input(사용자 입력) 수신: chat_input(채팅 입력) 또는 suggestion button(제안 버튼) ---
    prompt = st.chat_input("EV-Charge AI에게 충전 관제 데이터에 대해 물어보세요...")  # chat_input(채팅 입력) widget(위젯)
    if clicked_prompt:  # suggestion button(제안 버튼)이 클릭되었으면
        prompt = clicked_prompt  # clicked_prompt(클릭된 추천 질문)을 prompt(프롬프트)로 사용

    if prompt:  # prompt(프롬프트)가 존재하면 AI 응답 처리 시작
        # --- 새 prompt(프롬프트) 시 이전 SHAP chart(차트) session_state(세션 상태) 초기화 ---
        if "last_shap_chart" in st.session_state:  # last_shap_chart(최근 SHAP 차트) 키가 있으면
            del st.session_state["last_shap_chart"]  # 이전 SHAP chart(차트) 데이터 삭제
            
        # --- user message(사용자 메시지) chat bubble(말풍선) 표시 ---
        with st.chat_message("user"):  # user(사용자) role(역할) chat bubble(말풍선)
            st.markdown(prompt)  # user prompt(사용자 프롬프트) 마크다운 렌더링
        st.session_state.messages.append({"role": "user", "content": prompt})  # chat history(채팅 기록)에 user message(사용자 메시지) 추가

        # --- assistant(어시스턴트) response(응답) chat bubble(말풍선) 표시 ---
        with st.chat_message("assistant"):  # assistant(어시스턴트) role(역할) chat bubble(말풍선)
            message_placeholder = st.empty()  # empty(빈) placeholder(플레이스홀더) — streaming(스트리밍) 응답용
            
            try:  # AI API 호출 try 블록 시작
                if current_provider == "Gemini (Google)":  # Gemini(제미니) provider(제공자)인 경우
                    # --- Gemini(제미니) GenerativeModel(생성 모델) 초기화 — Function Calling(함수 호출) tools(도구) 등록 ---
                    selected_model_name = st.session_state.get("gemini_selected_model", "gemini-1.5-flash")  # session_state(세션 상태)에서 선택된 model(모델)명 조회
                    chat_model = genai.GenerativeModel(  # GenerativeModel(생성 모델) 객체 생성
                        model_name=selected_model_name,  # model(모델)명
                        system_instruction=system_instruction,  # system_instruction(시스템 지침)
                        tools=[get_simulation_result, get_shap_analysis, run_dr_simulation, get_top_regions, generate_report_from_conversation]  # Function Calling(함수 호출) tools(도구) list(리스트)
                    )
                    
                    # --- Gemini(제미니) chat history(채팅 기록) format(형식) 변환 ---
                    formatted_history = []  # formatted_history(변환된 기록) list(리스트) 초기화
                    for msg in st.session_state.messages[:-1]:  # messages(메시지) list(리스트) 순회 (현재 prompt(프롬프트) 제외)
                        gemini_role = "user" if msg["role"] == "user" else "model"  # Gemini(제미니) role(역할) 매핑: user/model
                        formatted_history.append({  # history(기록) entry(항목) 추가
                            "role": gemini_role,  # role(역할)
                            "parts": [msg["content"]]  # parts(파트) — text content(텍스트 콘텐츠)
                        })
                    
                    chat = chat_model.start_chat(history=formatted_history)  # chat(채팅) session(세션) 시작 — history(기록) 주입
                    
                    # --- Gemini(제미니) API response(응답) fetch(가져오기) ---
                    with st.spinner("AI 분석가가 데이터를 분석 중입니다..."):  # spinner(로딩 스피너) — 분석 중 표시
                        response = chat.send_message(prompt)  # send_message(메시지 전송) — user prompt(사용자 프롬프트) 전송
                        
                        # --- Function Calling(함수 호출) loop(루프) — chained(연쇄) tool call(도구 호출) 처리 ---
                        max_iterations = 8  # max_iterations(최대 반복 횟수) — 무한 loop(루프) 방지
                        iteration = 0  # iteration(반복) 카운터 초기화
                        while iteration < max_iterations:  # max_iterations(최대 반복) 이내 loop(루프)
                            iteration += 1  # iteration(반복) 카운터 증가
                            has_function_call = False  # has_function_call(함수 호출 존재) flag(플래그) 초기화
                            if response.candidates and response.candidates[0].content.parts:  # response(응답)에 parts(파트)가 있으면
                                parts = response.candidates[0].content.parts  # response(응답) parts(파트) 추출
                                for part in parts:  # parts(파트) 순회
                                    if hasattr(part, "function_call") and part.function_call:  # function_call(함수 호출) part(파트)인 경우
                                        has_function_call = True  # has_function_call(함수 호출 존재) flag(플래그) 설정
                                        name = part.function_call.name  # function(함수) name(이름)
                                        args = part.function_call.args  # function(함수) arguments(인자) dict(딕셔너리)
                                        
                                        # --- function(함수) name(이름)별 Python function(함수) dispatch(디스패치) ---
                                        if name == "get_simulation_result":  # 충전기 증설 시뮬레이션
                                            region = args.get("region", "")  # region(지역명) 인자
                                            count = int(args.get("count", 10))  # count(대수) 인자
                                            power_type = args.get("power_type", "급속")  # power_type(충전기 타입) 인자
                                            sim_res = get_simulation_result(region, count, power_type)  # 시뮬레이션 function(함수) 실행
                                        elif name == "get_shap_analysis":  # SHAP(설명 가능 AI) 분석
                                            region = args.get("region", "")  # region(지역명) 인자
                                            sim_res = get_shap_analysis(region)  # SHAP function(함수) 실행
                                        elif name == "run_dr_simulation":  # DR(수요반응) 시뮬레이션
                                            region = args.get("region", "")  # region(지역명) 인자
                                            intervention_type = args.get("intervention_type", "V2G_Peak_Shaving")  # intervention_type(정책 타입) 인자
                                            sim_res = run_dr_simulation(region, intervention_type)  # DR function(함수) 실행
                                        elif name == "get_top_regions":  # 상위 N개 지역 조회
                                            n = int(args.get("n", 10))  # n(개수) 인자
                                            metric = args.get("metric", "전력_부하지수")  # metric(지표) 인자
                                            sim_res = get_top_regions(n, metric)  # top regions function(함수) 실행
                                        elif name == "generate_report_from_conversation":  # PDF(포터블 문서) 보고서 생성
                                            region = args.get("region", "")  # region(지역명) 인자
                                            sim_res = generate_report_from_conversation(region)  # PDF generation(생성) function(함수) 실행
                                        else:  # 알 수 없는 function(함수) name(이름)
                                            sim_res = "알 수 없는 함수 호출입니다."  # error(에러) 메시지
                                            
                                        # --- function response(함수 응답)를 Gemini(제미니) LLM(대규모 언어 모델)에 feedback(피드백) ---
                                        try:  # Part.from_function_response(함수 응답 파트) 생성 try 블록
                                            func_response_part = genai.types.Part.from_function_response(  # function_response(함수 응답) Part(파트) 생성
                                                name=name,  # function(함수) name(이름)
                                                response={"result": sim_res}  # function(함수) 실행 result(결과)
                                            )
                                        except Exception:  # Part.from_function_response(함수 응답 파트) 생성 실패 시
                                            func_response_part = {  # dict(딕셔너리) fallback(대체) format(형식)
                                                "function_response": {
                                                    "name": name,
                                                    "response": {"result": sim_res}
                                                }
                                            }
                                        
                                        response = chat.send_message(func_response_part)  # function response(함수 응답) 전송 — LLM(대규모 언어 모델) 재추론
                                        break  # iteration(반복)당 하나의 function call(함수 호출)만 처리 후 re-check(재확인)
                            
                            if not has_function_call:  # function call(함수 호출)이 없으면
                                break  # Function Calling(함수 호출) loop(루프) 종료
                                        
                        # --- final(최종) text response(텍스트 응답) 추출 ---
                        try:  # response.text(응답 텍스트) 추출 try 블록
                            full_response = response.text  # Gemini(제미니) 최종 text response(텍스트 응답)
                        except Exception:  # text(텍스트) 추출 실패 시
                            full_response = "응답 처리 중 오류가 발생했습니다. 다시 시도해 주세요."  # fallback(대체) error(에러) 메시지
                else:  # Groq (Llama 3) provider(제공자)인 경우 — OpenAI-compatible(호환) API(애플리케이션 프로그래밍 인터페이스)
                    # --- OpenAI-compatible(호환) tools(도구) schema(스키마) 정의 ---
                    openai_tools = [  # OpenAI function calling(함수 호출) tools(도구) schema(스키마) list(리스트)
                        {  # tool(도구) 1: get_simulation_result(충전기 증설 시뮬레이션)
                            "type": "function",
                            "function": {
                                "name": "get_simulation_result",
                                "description": "지정된 수도권 행정구역에 특정 충전기 타입과 대수를 증설했을 때의 예측 전력 부하지수 감소량 및 전/후 비교 결과를 조회하거나 실시간 계산합니다.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "region": {"type": "string", "description": "분석할 행정구역명 (예: '안양시', '수원시 동안구' 등)"},
                                        "count": {"type": "integer", "description": "설치할 충전기 대수 (예: 10)"},
                                        "power_type": {"type": "string", "description": "충전기 타입 ('급속' 또는 '완속')"}
                                    },
                                    "required": ["region", "count"]
                                }
                            }
                        },
                        {  # tool(도구) 2: get_shap_analysis(SHAP 기여도 분석)
                            "type": "function",
                            "function": {
                                "name": "get_shap_analysis",
                                "description": "지정된 수도권 행정구역의 부하 예측 결과에 대해 각 피처(인프라, 전기차수, 용량 등)가 미친 기여도(SHAP)를 분석합니다.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "region": {"type": "string", "description": "분석할 행정구역명 (예: '안양시' 등)"}
                                    },
                                    "required": ["region"]
                                }
                            }
                        },
                        {  # tool(도구) 3: run_dr_simulation(수요반응 DR 시뮬레이션)
                            "type": "function",
                            "function": {
                                "name": "run_dr_simulation",
                                "description": "지정된 수도권 행정구역에 스마트 그리드 수요반응(DR) 또는 피크 컷 정책을 도입했을 때의 예측 전력 부하지수 완화율 및 과부하 지연 효과를 시뮬레이션합니다.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "region": {"type": "string", "description": "분석할 행정구역명 (예: '안양시' 등)"},
                                        "intervention_type": {"type": "string", "description": "적용할 정책 개입 타입 ('V2G_Peak_Shaving' 또는 'Smart_Charging_50')"}
                                    },
                                    "required": ["region"]
                                }
                            }
                        },
                        {  # tool(도구) 4: get_top_regions(상위 N개 고위험 지역 조회)
                            "type": "function",
                            "function": {
                                "name": "get_top_regions",
                                "description": "지정된 수도권 내에서 전력 부하지수 또는 인프라 부하지수가 가장 높은 상위 N개 지역의 상세 통계 테이블을 조회합니다.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "n": {"type": "integer", "description": "조회할 상위 지역 개수 (예: 10)"},
                                        "metric": {"type": "string", "description": "정렬 기준이 되는 지표 ('전력_부하지수' 또는 '인프라_부하지수')"}
                                    },
                                    "required": []
                                }
                            }
                        },
                        {  # tool(도구) 5: generate_report_from_conversation(PDF 보고서 생성)
                            "type": "function",
                            "function": {
                                "name": "generate_report_from_conversation",
                                "description": "사용자가 특정 행정구역 분석 보고서 출력을 원할 때 호출되어, 해당 지역의 상세 통계, 예측 부하지수 및 정책 제안이 포함된 맞춤형 PDF 보고서를 생성하고 다운로드 링크를 준비합니다.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "region": {"type": "string", "description": "보고서를 발간할 행정구역명 (예: '안양시', '부천시' 등)"}
                                    },
                                    "required": ["region"]
                                }
                            }
                        }
                    ]  # openai_tools(도구 스키마) list(리스트) 끝

                    # --- OpenAI-compatible(호환) messages(메시지) list(리스트) 구성 ---
                    openai_messages = [  # openai_messages(메시지) list(리스트) 초기화
                        {"role": "system", "content": system_instruction}  # system(시스템) role(역할) — system_instruction(시스템 지침)
                    ]
                    # --- chat history(채팅 기록) append(추가) ---
                    for msg in st.session_state.messages[:-1]:  # messages(메시지) list(리스트) 순회 (현재 prompt(프롬프트) 제외)
                        openai_role = "user" if msg["role"] == "user" else "assistant"  # OpenAI role(역할) 매핑: user/assistant
                        openai_messages.append({  # history(기록) entry(항목) 추가
                            "role": openai_role,  # role(역할)
                            "content": msg["content"]  # text content(텍스트 콘텐츠)
                        })
                    # --- current user prompt(현재 사용자 프롬프트) 추가 ---
                    openai_messages.append({  # current prompt(현재 프롬프트) entry(항목) 추가
                        "role": "user",  # user(사용자) role(역할)
                        "content": prompt  # user prompt(사용자 프롬프트) text(텍스트)
                    })

                    with st.spinner("AI 분석가가 데이터를 분석 중입니다..."):  # spinner(로딩 스피너) — Groq(그록) API 호출 중 표시
                        content, tool_calls = call_openai_compatible_api(  # Groq(그록) API 1차 호출 — content(텍스트) + tool_calls(함수 호출) 반환
                            current_provider, active_key, openai_messages, openai_tools  # provider(제공자), key(키), messages(메시지), tools(도구)
                        )
                        
                        if tool_calls:  # tool_calls(함수 호출)이 존재하면
                            # --- assistant(어시스턴트) tool_call(함수 호출) message(메시지)를 context(컨텍스트)에 추가 ---
                            openai_messages.append({  # assistant tool_call(함수 호출) entry(항목) 추가
                                "role": "assistant",  # assistant(어시스턴트) role(역할)
                                "content": content,  # assistant(어시스턴트) text content(텍스트) (있을 경우)
                                "tool_calls": tool_calls  # tool_calls(함수 호출) list(리스트)
                            })
                            
                            # --- 각 tool_call(함수 호출) 순회 처리 ---
                            for tool_call in tool_calls:  # tool_calls(함수 호출) list(리스트) 순회
                                tc_id = tool_call.get("id")  # tool_call_id(함수 호출 ID)
                                func_info = tool_call.get("function")  # function(함수) info(정보) dict(딕셔너리)
                                name = func_info.get("name")  # function(함수) name(이름)
                                args = json.loads(func_info.get("arguments", "{}"))  # arguments(인자) JSON(자바스크립트 객체 표기법) 파싱
                                
                                # --- function(함수) name(이름)별 Python function(함수) dispatch(디스패치) ---
                                if name == "get_simulation_result":  # 충전기 증설 시뮬레이션
                                    region = args.get("region", "")  # region(지역명) 인자
                                    count = int(args.get("count", 10))  # count(대수) 인자
                                    power_type = args.get("power_type", "급속")  # power_type(충전기 타입) 인자
                                    sim_res = get_simulation_result(region, count, power_type)  # 시뮬레이션 function(함수) 실행
                                elif name == "get_shap_analysis":  # SHAP(설명 가능 AI) 분석
                                    region = args.get("region", "")  # region(지역명) 인자
                                    sim_res = get_shap_analysis(region)  # SHAP function(함수) 실행
                                elif name == "run_dr_simulation":  # DR(수요반응) 시뮬레이션
                                    region = args.get("region", "")  # region(지역명) 인자
                                    intervention_type = args.get("intervention_type", "V2G_Peak_Shaving")  # intervention_type(정책 타입) 인자
                                    sim_res = run_dr_simulation(region, intervention_type)  # DR function(함수) 실행
                                elif name == "get_top_regions":  # 상위 N개 지역 조회
                                    n = int(args.get("n", 10))  # n(개수) 인자
                                    metric = args.get("metric", "전력_부하지수")  # metric(지표) 인자
                                    sim_res = get_top_regions(n, metric)  # top regions function(함수) 실행
                                elif name == "generate_report_from_conversation":  # PDF(포터블 문서) 보고서 생성
                                    region = args.get("region", "")  # region(지역명) 인자
                                    sim_res = generate_report_from_conversation(region)  # PDF generation(생성) function(함수) 실행
                                else:  # 알 수 없는 function(함수) name(이름)
                                    sim_res = "알 수 없는 함수 호출입니다."  # error(에러) 메시지
                                    
                                # --- tool response(도구 응답) message(메시지) append(추가) ---
                                openai_messages.append({  # tool response(도구 응답) entry(항목) 추가
                                    "role": "tool",  # tool(도구) role(역할)
                                    "tool_call_id": tc_id,  # tool_call_id(함수 호출 ID) — assistant tool_call(함수 호출)과 매칭
                                    "name": name,  # function(함수) name(이름)
                                    "content": sim_res  # function(함수) 실행 result(결과) text(텍스트)
                                })
                                
                            # --- 2차 API 호출 — function result(함수 결과) 반영 후 final(최종) text response(텍스트 응답) 획득 ---
                            content, _ = call_openai_compatible_api(  # Groq(그록) API 2차 호출 (tools(도구) 없이)
                                current_provider, active_key, openai_messages  # provider(제공자), key(키), messages(메시지)
                            )
                            
                        full_response = content  # full_response(전체 응답) = Groq(그록) 최종 content(텍스트)
                    
                message_placeholder.markdown(full_response)  # assistant(어시스턴트) response(응답) 마크다운 렌더링
                
                # --- assistant(어시스턴트) response(응답)를 chat history(채팅 기록)에 append(추가) — optional(선택) PDF(포터블 문서) metadata(메타데이터) 포함 ---
                msg_data = {"role": "assistant", "content": full_response}  # msg_data(메시지 데이터) dict(딕셔너리) 생성
                if "pending_pdf_bytes" in st.session_state and "pending_pdf_region" in st.session_state:  # pending PDF(대기 중 PDF) metadata(메타데이터) 존재 시
                    msg_data["pdf_bytes"] = st.session_state.pop("pending_pdf_bytes")  # PDF(포터블 문서) bytes(바이트) pop(추출) 후 msg_data(메시지 데이터)에 추가
                    msg_data["pdf_region"] = st.session_state.pop("pending_pdf_region")  # PDF(포터블 문서) region(지역명) pop(추출) 후 msg_data(메시지 데이터)에 추가
                st.session_state.messages.append(msg_data)  # chat history(채팅 기록)에 assistant message(어시스턴트 메시지) append(추가)
                
                # --- 새로 생성된 PDF(포터블 문서) download button(다운로드 버튼) 즉시 렌더링 ---
                if msg_data.get("pdf_bytes") and msg_data.get("pdf_region"):  # PDF(포터블 문서) metadata(메타데이터)가 있으면
                    st.download_button(  # download_button(다운로드 버튼) — 새 PDF(포터블 문서) 다운로드 UI
                        label=f"⬇️ {msg_data['pdf_region']} 관제 분석 보고서 다운로드 (PDF)",
                        data=msg_data["pdf_bytes"],  # PDF(포터블 문서) 바이트 데이터
                        file_name=f"{msg_data['pdf_region']}_EV_charge_report.pdf",  # download(다운로드) 파일명
                        mime="application/pdf",  # MIME(미디어 타입) — PDF(포터블 문서)
                        key=f"dl_btn_new_{msg_data['pdf_region']}"  # widget(위젯) unique key(고유 키)
                    )
                
            except Exception as e:  # AI API 호출 중 예외 포착
                err_str = str(e)  # exception(예외) message(메시지) 문자열 변환
                # --- API key(키) 등 sensitive(민감) 정보 masking(마스킹) — error message(에러 메시지) frontend(프론트엔드) 노출 방지 ---
                import re  # re(정규표현식) 모듈 import(임포트)
                clean_err = re.sub(r'(Bearer\s+)[a-zA-Z0-9_\-\.\/]+', r'\1[MASKED_API_KEY]', err_str)  # Bearer token(베어러 토큰) masking(마스킹)
                clean_err = re.sub(r'(gsk_[a-zA-Z0-9_\-]+|sk\-[a-zA-Z0-9_\-]+)', '[MASKED_API_KEY]', clean_err)  # Groq/OpenAI key(키) prefix(접두사) masking(마스킹)
                
                if "429" in clean_err or "quota" in clean_err.lower() or "ResourceExhausted" in clean_err:  # rate limit(호출 한도) / quota(할당량) 초과 error(에러)
                    st.error(  # error(에러) UI — quota(할당량) 초과 안내
                        f"⚠️ **AI 서비스 API 호출 한도(Quota/Rate Limit)를 초과했습니다.**\n\n"
                        f"현재 사용 중인 {current_provider} API Key의 호출 제한에 도달했거나 할당량이 부족합니다. "
                        f"잠시 대기하신 후 다시 시도해 주세요.\n\n"
                        f"*상세 에러: {clean_err}*"
                    )
                else:  # 그 외 일반 error(에러)
                    st.error(f"⚠️ **{current_provider} 호출 중 오류가 발생했습니다:** {clean_err}")  # error(에러) UI — 일반 오류 안내
                
        # --- rerun(재실행) — chat display(채팅 표시) 업데이트 (PDF(포터블 문서) 보고서 download button(다운로드 버튼)은 상단 generated_reports(생성된 보고서) section(섹션)에서 표시) ---
        st.rerun()  # st.rerun(재실행) — chat history(채팅 기록) UI 갱신

    # --- interactive(인터랙티브) Plotly(플로틀리) SHAP chart(차트) 렌더링 — session_state(세션 상태)에 last_shap_chart(최근 SHAP 차트) 데이터 존재 시 ---
    if st.session_state.get("last_shap_chart"):  # last_shap_chart(최근 SHAP 차트) session_state(세션 상태) 존재 시
        chart_info = st.session_state["last_shap_chart"]  # chart_info(차트 정보) dict(딕셔너리) 조회
        r_name = chart_info["region"]  # chart(차트) 대상 region(지역명)
        usage = chart_info["usage"]  # chart(차트) 대상 usage(용도)
        features = chart_info["features"]  # feature(피처)명 list(리스트)
        values = chart_info["values"]  # SHAP value(값) list(리스트)
        
        import plotly.express as px  # plotly.express(플로틀리 익스프레스) 차트 라이브러리 import(임포트)
        import pandas as pd  # pandas(판다스) DataFrame(데이터프레임) 처리 라이브러리 import(임포트)
        
        df_shap = pd.DataFrame({  # SHAP(설명 가능 AI) chart(차트)용 DataFrame(데이터프레임) 생성
            "변수명": features,  # feature(피처)명 column(열)
            "기여도 (SHAP)": values,  # SHAP value(값) column(열)
            "영향": ["상승 기여" if v >= 0 else "감소 기여" for v in values]  # 영향 direction(방향) column(열) — list comprehension(리스트 컴프리헨션)
        })
        
        df_shap["abs_val"] = df_shap["기여도 (SHAP)"].abs()  # abs_val(절대값) column(열) 추가 — 정렬용
        df_shap = df_shap.sort_values("abs_val", ascending=True).tail(8)  # abs_val(절대값) 오름차순 정렬 후 Top 8 추출
        
        fig = px.bar(  # px.bar(수평/수직 막대 차트) 생성
            df_shap,  # chart(차트) DataFrame(데이터프레임)
            x="기여도 (SHAP)",  # x-axis(가로축) — SHAP value(값)
            y="변수명",  # y-axis(세로축) — feature(피처)명
            color="영향",  # color(색상) — 상승/감소 기여 direction(방향)
            orientation="h",  # orientation(방향) — horizontal(수평) 막대
            color_discrete_map={"상승 기여": "#FF5A5F", "감소 기여": "#3B82F6"},  # color map(색상 맵) — 상승=빨강, 감소=파랑
            title=f"📊 {r_name} ({usage}) 예측 변수 영향도 (SHAP Top 8)"  # chart(차트) title(제목)
        )
        fig.update_layout(  # chart(차트) layout(레이아웃) 설정
            margin=dict(l=20, r=20, t=40, b=20),  # margin(여백) — left/right/top/bottom
            height=300  # chart(차트) height(높이) 300px
        )
        st.plotly_chart(fig, use_container_width=True)  # plotly_chart(플로틀리 차트) Streamlit(스트림릿) widget(위젯) 렌더링
        
        if st.button("❌ 차트 닫기", use_container_width=True):  # chart(차트) 닫기 button(버튼) 클릭 시
            del st.session_state["last_shap_chart"]  # last_shap_chart(최근 SHAP 차트) session_state(세션 상태) 삭제
            st.rerun()  # st.rerun(재실행) — chart(차트) UI 제거
