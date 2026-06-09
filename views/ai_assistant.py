import streamlit as st
import google.generativeai as genai
import json
import gc

@st.cache_data(show_spinner=False, ttl=600)
def get_simulation_result(region: str, count: int, power_type: str = "급속") -> str:
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
    import numpy as np
    import pandas as pd
    from utils.optimization import calculate_single_region_trajectory

    final_data = st.session_state.get("final_data")
    if final_data is None:
        return "시스템 에러: 데이터가 세션에 존재하지 않습니다."

    # 1. 지역명 매칭
    matched = final_data[final_data["지역"].str.contains(region, case=False, na=False)]
    if matched.empty:
        matched = final_data[final_data["시군구"].str.contains(region, case=False, na=False)]
        if matched.empty:
            return f"검색된 지역 '{region}'을 찾을 수 없습니다. 경기 안양시 동안구 등 수도권 내 정확한 행정구역명을 입력해 주세요."

    # 용량 매핑: 급속 100kW, 완속 7kW 기준
    added_kw = count * (100.0 if power_type == "급속" else 7.0)
    critical_threshold = float(final_data["전력_부하지수"].quantile(0.8))

    lines = []
    lines.append(f"### 🔮 {region} 지역 충전기 증설 시뮬레이션 결과")
    lines.append(f"- **증설 정책**: {power_type} 충전소 {count}대 증설 (공급 용량: +{added_kw:,.1f} kW)")
    lines.append(f"- **전력 부하 임계치**: {critical_threshold:,.2f} (상위 20% 고위험 기준선)\n")

    # 매칭된 각 세부 구군별로 루프 실행
    for idx, row in matched.iterrows():
        r_name = row["지역"]
        usage = row["용도"]
        before_load = float(row["전력_부하지수"])
        base_load = float(row["총_전력판매량"])
        capacity = float(row["총용량_kW"])

        after_load = base_load / (capacity + added_kw) if (capacity + added_kw) > 0 else 0
        reduction_pct = (before_load - after_load) / before_load * 100 if before_load > 0 else 0

        # 생존 분석 (미래 과부하 도달 예측) - 기본 성장률은 5%로 설정
        traj_df, overload_before, overload_after = calculate_single_region_trajectory(
            base_load, capacity, 0.05, added_kw, critical_threshold
        )
        
        delay_years = overload_after - overload_before
        delay_text = f"+{delay_years}년 지연" if delay_years > 0 else "변동 없음"
        before_txt = f"{overload_before}년 뒤" if overload_before < 15 else "안전 (15년+)"
        after_txt = f"{overload_after}년 뒤" if overload_after < 15 else "안전 (15년+)"

        lines.append(f"#### 📍 {r_name} ({usage} 기준)")
        lines.append(f"  - **전력 부하지수**: {before_load:,.2f} ➡️ **{after_load:,.2f}** ({reduction_pct:+.1f}% 감소)")
        lines.append(f"  - **과부하 도달 시점**: {before_txt} ➡️ **{after_txt}** (지연 효과: {delay_text})")
        lines.append("")

    # 가비지 컬렉션 강제화로 즉시 메모리 반환
    gc.collect()

    return "\n".join(lines)

@st.cache_data(show_spinner=False, ttl=600)
def get_shap_analysis(region: str) -> str:
    """
    지정된 수도권 행정구역의 부하 예측 결과에 대해 
    각 피처(인프라, 전기차수, 용량 등)가 미친 긍정/부정적 기여도(SHAP)를 분석합니다.
    
    Args:
        region: 분석할 행정구역명 (예: '안양시 동안구', '안양시', '수원시 동안구' 등)
    Returns:
        기여도가 높은 주요 변수 분석 및 기여도 요약 텍스트
    """
    import numpy as np
    import pandas as pd
    import gc
    from utils.visualizations import get_cached_local_shap
    
    final_data = st.session_state.get("final_data")
    model_state = st.session_state.get("model_state")
    if final_data is None or model_state is None:
        return "시스템 에러: 모델 상태 또는 데이터가 존재하지 않습니다."
        
    matched = final_data[final_data["지역"].str.contains(region, case=False, na=False)]
    if matched.empty:
        matched = final_data[final_data["시군구"].str.contains(region, case=False, na=False)]
        if matched.empty:
            return f"검색된 지역 '{region}'을 찾을 수 없습니다. 경기 안양시 동안구 등 수도권 내 정확한 행정구역명을 입력해 주세요."
            
    best_model = model_state["models"][model_state["best_name"]]
    X_all = model_state["X"]
    
    local_row = matched[matched["용도"] == "자가용"]
    if local_row.empty:
        local_row = matched.head(1)
    
    r_name = local_row.iloc[0]["지역"]
    usage = local_row.iloc[0]["용도"]
    
    feature_cols = model_state["feature_columns"]
    local_x = X_all[(final_data["지역"] == r_name) & (final_data["용도"] == usage)].head(1)
    if local_x.empty:
        return f"'{r_name} ({usage})' 지역에 대한 피처 데이터를 찾지 못해 SHAP 분석을 수행할 수 없습니다."
        
    sample = X_all.sample(min(60, len(X_all)), random_state=42)
    
    try:
        base_val, vals = get_cached_local_shap(model_state["best_name"], best_model, sample, local_x)
        
        contributions = []
        for col, val in zip(feature_cols, vals):
            contributions.append({
                "Feature": col,
                "SHAP_Value": float(val)
            })
            
        df_contrib = pd.DataFrame(contributions)
        df_contrib["Abs_SHAP"] = df_contrib["SHAP_Value"].abs()
        df_contrib = df_contrib.sort_values("Abs_SHAP", ascending=False)
        
        lines = []
        lines.append(f"### 🧠 {r_name} ({usage}) 부하 예측 SHAP 기여도 분석")
        lines.append(f"- **기본 기대 예측값(Base Value)**: {base_val:.2f}")
        lines.append(f"- **최종 모델 예측 부하지수**: {local_row.iloc[0]['전력_부하지수']:.2f}\n")
        lines.append("#### 📊 주요 피처별 영향력 분석 (Top 5)")
        
        for _, r in df_contrib.head(5).iterrows():
            f_name = r["Feature"]
            s_val = r["SHAP_Value"]
            direction = "🔺 증가 기여" if s_val >= 0 else "🔻 감소 기여"
            lines.append(f"  - **{f_name}**: {s_val:+.4f} ({direction})")
            
        st.session_state["last_shap_chart"] = {
            "region": r_name,
            "usage": usage,
            "features": df_contrib["Feature"].tolist(),
            "values": df_contrib["SHAP_Value"].tolist()
        }
        
        gc.collect()
        return "\n".join(lines)
        
    except Exception as e:
        return f"SHAP 계산 중 에러가 발생했습니다: {e}"

@st.cache_data(show_spinner=False, ttl=600)
def run_dr_simulation(region: str, intervention_type: str = "V2G_Peak_Shaving") -> str:
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
    import numpy as np
    import pandas as pd
    import gc
    from utils.optimization import calculate_single_region_trajectory

    final_data = st.session_state.get("final_data")
    if final_data is None:
        return "시스템 에러: 데이터가 세션에 존재하지 않습니다."

    matched = final_data[final_data["지역"].str.contains(region, case=False, na=False)]
    if matched.empty:
        matched = final_data[final_data["시군구"].str.contains(region, case=False, na=False)]
        if matched.empty:
            return f"검색된 지역 '{region}'을 찾을 수 없습니다. 경기 안양시 동안구 등 수도권 내 정확한 행정구역명을 입력해 주세요."

    if "V2G" in intervention_type:
        reduction_multiplier = 0.70
        policy_name = "V2G 양방향 충방전 Peak Shaving (피크 부하 30% 감축)"
    else:
        reduction_multiplier = 0.80
        policy_name = "Smart Charging 50% 분배제한 (피크 부하 20% 감축)"

    critical_threshold = float(final_data["전력_부하지수"].quantile(0.8))

    lines = []
    lines.append(f"### 🔋 {region} 스마트 그리드 수요반응(DR) 시뮬레이션 결과")
    lines.append(f"- **적용 정책**: {policy_name}")
    lines.append(f"- **전력 부하 임계치**: {critical_threshold:,.2f} (상위 20% 고위험 기준선)\n")

    for idx, row in matched.iterrows():
        r_name = row["지역"]
        usage = row["용도"]
        before_load = float(row["전력_부하지수"])
        base_load = float(row["총_전력판매량"])
        capacity = float(row["총용량_kW"])

        sim_load = base_load * reduction_multiplier
        after_load = sim_load / capacity if capacity > 0 else 0
        reduction_pct = (before_load - after_load) / before_load * 100 if before_load > 0 else 0

        traj_df, overload_before, overload_after = calculate_single_region_trajectory(
            sim_load, capacity, 0.05, 0.0, critical_threshold
        )
        _, overload_before_orig, _ = calculate_single_region_trajectory(
            base_load, capacity, 0.05, 0.0, critical_threshold
        )

        delay_years = overload_after - overload_before_orig
        delay_text = f"+{delay_years}년 지연" if delay_years > 0 else "변동 없음"
        before_txt = f"{overload_before_orig}년 뒤" if overload_before_orig < 15 else "안전 (15년+)"
        after_txt = f"{overload_after}년 뒤" if overload_after < 15 else "안전 (15년+)"

        lines.append(f"#### 📍 {r_name} ({usage} 기준)")
        lines.append(f"  - **전력 부하지수**: {before_load:,.2f} ➡️ **{after_load:,.2f}** ({reduction_pct:+.1f}% 감소)")
        lines.append(f"  - **과부하 도달 시점**: {before_txt} ➡️ **{after_txt}** (지연 효과: {delay_text})")
        lines.append("")

    gc.collect()
    return "\n".join(lines)

@st.cache_data(show_spinner=False, ttl=600)
def get_top_regions(n: int = 10, metric: str = "전력_부하지수") -> str:
    """
    수도권 내에서 전력 부하지수 또는 인프라 부하지수가 가장 높은 상위 N개 지역의 상세 통계 테이블을 조회합니다.
    
    Args:
        n: 조회할 상위 지역 개수 (예: 10)
        metric: 정렬 기준이 되는 지표 ('전력_부하지수' 또는 '인프라_부하지수')
    Returns:
        상위 N개 지역의 통계 정보가 담긴 마크다운 표
    """
    import pandas as pd
    import gc

    final_data = st.session_state.get("final_data")
    if final_data is None:
        return "시스템 에러: 데이터가 세션에 존재하지 않습니다."

    # 지표명 매핑 (영문/국문 허용)
    if "인프라" in metric or "infra" in metric.lower():
        sort_col = "인프라_부하지수"
    else:
        sort_col = "전력_부하지수"

    if sort_col not in final_data.columns:
        return f"에러: 지표 '{sort_col}'를 데이터셋에서 찾을 수 없습니다."

    # 상위 N개 정렬
    top_df = final_data.sort_values(sort_col, ascending=False).head(n)
    
    lines = []
    lines.append(f"### 📊 수도권 {sort_col} 상위 {n}개 우려지역")
    lines.append("| 순위 | 지역명 | 용도 | 전력 부하지수 | 인프라 부하지수 | 전체 충전기 대수 | 총용량 (kW) |")
    lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: |")
    
    for idx, (_, row) in enumerate(top_df.iterrows(), 1):
        r_name = row["지역"]
        usage = row["용도"]
        p_load = float(row["전력_부하지수"])
        i_load = float(row["인프라_부하지수"])
        chargers = int(row["전체_충전기대수"])
        capacity = float(row["총용량_kW"])
        lines.append(f"| {idx} | {r_name} | {usage} | {p_load:,.2f} | {i_load:,.2f} | {chargers:,}대 | {capacity:,.1f} kW |")

    gc.collect()
    return "\n".join(lines)

def generate_report_from_conversation(region: str) -> str:
    """
    사용자가 특정 행정구역 분석 보고서 출력을 원할 때 호출되어, 해당 지역의 상세 통계, 예측 부하지수 및 정책 제안이 포함된 맞춤형 PDF 보고서를 생성하고 다운로드 링크를 준비합니다.
    
    Args:
        region: 보고서를 발간할 행정구역명 (예: '안양시', '부천시' 등)
    Returns:
        보고서 PDF 파일 생성 완료 메시지 및 안내 문구
    """
    from utils.pdf_generator import generate_regional_report_pdf
    
    final_data = st.session_state.get("final_data")
    hourly_data = st.session_state.get("hourly_data")
    if final_data is None:
        return "시스템 에러: 데이터를 로드하지 못했습니다."
        
    try:
        pdf_bytes = generate_regional_report_pdf(region, final_data, hourly_data)
        st.session_state["pdf_report_bytes"] = pdf_bytes
        st.session_state["pdf_report_region"] = region
        st.session_state["pending_pdf_bytes"] = pdf_bytes
        st.session_state["pending_pdf_region"] = region
        
        # Save to global generated reports dict in session state
        if "generated_reports" not in st.session_state:
            st.session_state["generated_reports"] = {}
        st.session_state["generated_reports"][region] = pdf_bytes
        
        return f"성공적으로 {region} 지역의 맞춤형 관제 분석 보고서 PDF(3페이지 분량)가 생성되었습니다. 아래 다운로드 버튼을 확인하십시오."
    except Exception as e:
        return f"보고서 PDF 생성 실패: {str(e)}"

def call_openai_compatible_api(provider: str, api_key: str, messages: list, tools: list = None) -> tuple:
    """
    Groq의 OpenAI 호환 API를 호출합니다.
    Returns:
        (assistant_message_content, tool_calls_list)
    """
    import requests
    import json
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    model = "llama-3.3-70b-versatile"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3
    }
    
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
        
    res = requests.post(url, headers=headers, json=payload, timeout=30)
    res.raise_for_status()
    res_json = res.json()
    
    choice = res_json["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    
    return content, tool_calls

def get_gemini_client():
    # 1. Check secrets first
    api_key = st.secrets.get("GEMINI_API_KEY")
    
    # 2. Check session state (fallback from user input)
    if not api_key:
        api_key = st.session_state.get("user_gemini_api_key")
        
    if not api_key:
        return None
        
    try:
        genai.configure(api_key=api_key)
        
        # Programmatically discover available models for the given API key
        available_model_names = []
        try:
            for m in genai.list_models():
                if hasattr(m, 'supported_generation_methods') and "generateContent" in m.supported_generation_methods:
                    available_model_names.append(m.name)
        except Exception:
            pass
            
        candidate_models = [
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro",
            "gemini-1.0-pro",
        ]
        
        selected_model_name = None
        for candidate in candidate_models:
            full_candidate = candidate if candidate.startswith("models/") else f"models/{candidate}"
            if full_candidate in available_model_names:
                selected_model_name = candidate
                break
                
        if not selected_model_name:
            if available_model_names:
                selected_model_name = available_model_names[0].replace("models/", "")
            else:
                selected_model_name = "gemini-1.5-flash"
                
        st.session_state["gemini_selected_model"] = selected_model_name
        model = genai.GenerativeModel(selected_model_name)
        return model
    except Exception as e:
        st.error(f"Gemini API 설정 중 오류가 발생했습니다: {e}")
        return None

def build_system_context(filtered_data, model_state, control_mode, hw_data=None):
    """
    Builds a detailed JSON-like context text summarizing the current dashboard state
    to inject into the Gemini system instructions.
    """
    context = {
        "dashboard_mode": control_mode,
    }
    
    if control_mode == "도심 행정구역 관제" and filtered_data is not None:
        # Urban context aggregation
        total_districts = len(filtered_data)
        avg_power_load = float(filtered_data["전력_부하지수"].mean()) if "전력_부하지수" in filtered_data.columns else 0.0
        avg_infra_load = float(filtered_data["인프라_부하지수"].mean()) if "인프라_부하지수" in filtered_data.columns else 0.0
        
        # Identify TOP 3 high load areas
        top3 = []
        if "전력_부하지수" in filtered_data.columns:
            top3_df = filtered_data.sort_values("전력_부하지수", ascending=False).head(3)
            for _, row in top3_df.iterrows():
                top3.append({
                    "지역": row.get("지역", f"{row.get('시도', '')} {row.get('구군', '')}"),
                    "용도": row.get("용도", "N/A"),
                    "전력_부하지수": float(row.get("전력_부하지수", 0.0)),
                    "인프라_부하지수": float(row.get("인프라_부하지수", 0.0))
                })
        
        # Model evaluation information
        best_model_name = "N/A"
        test_rmse = 0.0
        if model_state and "best_name" in model_state:
            best_model_name = model_state["best_name"]
            metrics_df = model_state.get("metrics")
            if metrics_df is not None and not metrics_df.empty:
                test_rmse_row = metrics_df[(metrics_df["Model"] == best_model_name) & (metrics_df["Split"] == "Test")]
                if not test_rmse_row.empty:
                    test_rmse = float(test_rmse_row["RMSE"].values[0])
            
        context["urban_summary"] = {
            "총_필터링된_행정구역_수": total_districts,
            "평균_전력_부하지수": round(avg_power_load, 4),
            "평균_인프라_부하지수": round(avg_infra_load, 4),
            "TOP_3_고위험_우려지역": top3,
            "최적_예측_모델": best_model_name,
            "예측_오차_RMSE": round(test_rmse, 4)
        }
        
    elif control_mode == "고속도로망 최적화" and hw_data is not None:
        # Highway context
        total_nodes = len(hw_data)
        avg_traffic = float(hw_data["교통량"].mean()) if "교통량" in hw_data.columns else 0.0
        context["highway_summary"] = {
            "총_고속도로_노드_수": total_nodes,
            "평균_교통량": round(avg_traffic, 2)
        }
        
    return json.dumps(context, ensure_ascii=False, indent=2)

def render_ai_assistant(filtered_data, model_state, control_mode, hw_data=None):
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
    """, unsafe_allow_html=True)

    # API configuration and keys retrieval
    current_provider = st.session_state.get("ai_provider", "Gemini (Google)")
    if current_provider not in ["Gemini (Google)", "Groq (Llama 3)"]:
        current_provider = "Gemini (Google)"
        st.session_state["ai_provider"] = current_provider
    
    # 1. Retrieve keys from st.secrets first (Backend only, never sent to frontend)
    gemini_secret_key = st.secrets.get("GEMINI_API_KEY")
    groq_secret_key = st.secrets.get("GROQ_API_KEY")
    
    # Check if we have key for the active provider
    active_key = None
    if current_provider == "Gemini (Google)":
        if gemini_secret_key:
            active_key = gemini_secret_key
        else:
            active_key = st.session_state.get("user_gemini_api_key")
    else: # Groq (Llama 3)
        if groq_secret_key:
            active_key = groq_secret_key
        else:
            active_key = st.session_state.get("user_groq_api_key")

    # Expand settings UI
    with st.expander("⚙️ AI 관제비서 API 및 엔진 설정", expanded=not active_key):
        provider_col, key_col = st.columns([1, 2])
        with provider_col:
            selected_prov = st.selectbox(
                "AI 서비스 제공자",
                ["Gemini (Google)", "Groq (Llama 3)"],
                index=["Gemini (Google)", "Groq (Llama 3)"].index(current_provider),
                key="ai_provider_input"
            )
            if selected_prov != current_provider:
                st.session_state["ai_provider"] = selected_prov
                st.rerun()
                
        with key_col:
            # Check if key is available in Secrets for the selected provider
            has_secret = False
            if selected_prov == "Gemini (Google)" and gemini_secret_key:
                has_secret = True
            elif selected_prov == "Groq (Llama 3)" and groq_secret_key:
                has_secret = True
                
            if has_secret:
                st.success("✅ **API Key가 시스템 설정(Secrets)에 안전하게 등록되어 있습니다.**")
                st.caption("보안을 위해 실제 API 키는 브라우저에 표시되거나 전송되지 않습니다.")
            else:
                # Show text input only if not available in Secrets, but do NOT populate value from secrets!
                if selected_prov == "Gemini (Google)":
                    gemini_key = st.text_input(
                        "Gemini API Key 입력 (임시)",
                        type="password",
                        placeholder="API Key를 입력하세요",
                        value=st.session_state.get("user_gemini_api_key", ""),
                        key="user_gemini_key_input_new"
                    )
                    if gemini_key and gemini_key != st.session_state.get("user_gemini_api_key"):
                        st.session_state["user_gemini_api_key"] = gemini_key
                        st.rerun()
                else: # Groq (Llama 3)
                    groq_key = st.text_input(
                        "Groq API Key 입력 (임시)",
                        type="password",
                        placeholder="API Key를 입력하세요",
                        value=st.session_state.get("user_groq_api_key", ""),
                        key="user_groq_key_input_new"
                    )
                    if groq_key and groq_key != st.session_state.get("user_groq_api_key"):
                        st.session_state["user_groq_api_key"] = groq_key
                        st.rerun()
                    
    if not active_key:
        st.warning(f"⚠️ **{current_provider} API 키가 등록되지 않았습니다.**")
        st.info("시작하려면 위의 'AI 관제비서 API 및 엔진 설정' 창을 열고 API 키를 입력해 주세요.")
        return
        
    # Configure Gemini only if chosen
    if current_provider == "Gemini (Google)":
        try:
            genai.configure(api_key=active_key)
            # Programmatically discover available models for the given API key
            available_model_names = []
            try:
                for m in genai.list_models():
                    if hasattr(m, 'supported_generation_methods') and "generateContent" in m.supported_generation_methods:
                        available_model_names.append(m.name)
            except Exception:
                pass
                
            candidate_models = [
                "gemini-1.5-flash",
                "gemini-1.5-flash-latest",
                "gemini-1.5-pro",
                "gemini-1.0-pro",
            ]
            
            selected_model_name = None
            for candidate in candidate_models:
                full_candidate = candidate if candidate.startswith("models/") else f"models/{candidate}"
                if full_candidate in available_model_names:
                    selected_model_name = candidate
                    break
                    
            if not selected_model_name:
                if available_model_names:
                    selected_model_name = available_model_names[0].replace("models/", "")
                else:
                    selected_model_name = "gemini-1.5-flash"
                    
            st.session_state["gemini_selected_model"] = selected_model_name
        except Exception as e:
            st.error(f"Gemini API 설정 중 오류가 발생했습니다: {e}")
            return

    # Info cards for quick overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="info-card">
                <h4>📊 실시간 데이터 연동</h4>
                <p>현재 필터링된 대시보드의 데이터를 AI가 실시간으로 분석 모델 지식과 융합하여 맞춤 답변합니다.</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="info-card">
                <h4>🔮 시뮬레이션 질의</h4>
                <p>"안양시에 급속 충전소 10대를 증설하면 부하가 얼마나 줄어들까?" 같은 시나리오 영향력을 문의해 보세요.</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="info-card">
                <h4>📋 맞춤형 보고서 생성</h4>
                <p>"안양시 보고서 만들어줘" 처럼 특정 지역 이름을 언급하면 AI가 3페이지 분량의 PDF 보고서를 자동 발간합니다.</p>
            </div>
        """, unsafe_allow_html=True)

    # 📋 발간된 보고서 다운로드 센터 (세션 내 생성된 보고서가 있을 때만 노출)
    if "generated_reports" in st.session_state and st.session_state["generated_reports"]:
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        st.markdown("### 📥 실시간 발간 완료된 맞춤형 보고서 목록")
        report_items = list(st.session_state["generated_reports"].items())
        cols = st.columns(min(len(report_items), 4))
        for col_idx, (region_name, r_bytes) in enumerate(report_items):
            with cols[col_idx % len(cols)]:
                st.download_button(
                    label=f"💾 {region_name} PDF 다운로드",
                    data=r_bytes,
                    file_name=f"{region_name}_EV_charge_report.pdf",
                    mime="application/pdf",
                    key=f"dl_btn_center_{region_name}_{col_idx}",
                    use_container_width=True
                )
        st.markdown("---")

    # Context Injection setup
    dashboard_context = build_system_context(filtered_data, model_state, control_mode, hw_data)
    
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
    """

    # Session State Chat History Init
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("pdf_bytes") and message.get("pdf_region"):
                st.download_button(
                    label=f"⬇️ {message['pdf_region']} 관제 분석 보고서 다운로드 (PDF)",
                    data=message["pdf_bytes"],
                    file_name=f"{message['pdf_region']}_EV_charge_report.pdf",
                    mime="application/pdf",
                    key=f"dl_btn_{idx}"
                )
            elif message["role"] == "assistant" and "generated_reports" in st.session_state:
                # Fallback: search for generated reports in the message content
                for region, pdf_bytes in st.session_state["generated_reports"].items():
                    short_region = region.replace("서울", "").replace("경기", "").replace("인천", "").strip()
                    if (region in message["content"] or short_region in message["content"]) and ("보고서" in message["content"] or "PDF" in message["content"]):
                        st.download_button(
                            label=f"📥 {region} 관제 분석 보고서 다운로드 (PDF)",
                            data=pdf_bytes,
                            file_name=f"{region}_EV_charge_report.pdf",
                            mime="application/pdf",
                            key=f"dl_btn_fallback_{idx}_{region}"
                        )
                        break

    # Suggestion Chips
    st.markdown("##### 💡 추천 질문 바로 하기")
    suggestion_cols = st.columns(3)
    
    suggestions = [
        "현재 수도권 내에서 가장 위험한 TOP 3 지역의 부하 지수를 비교해줘.",
        "학습된 예측 모델 중 최우수 성능 모델과 신뢰성에 대해 설명해줘.",
        "충전 부하가 가장 높은 병목 구간에 대한 단기 대책과 제안은 무엇인가요?"
    ]
    
    # Store clicked suggestion in session state
    clicked_prompt = None
    for i, suggestion in enumerate(suggestions):
        with suggestion_cols[i]:
            if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                clicked_prompt = suggestion

    # Accept user input (either from chat box or suggestion click)
    prompt = st.chat_input("EV-Charge AI에게 충전 관제 데이터에 대해 물어보세요...")
    if clicked_prompt:
        prompt = clicked_prompt

    if prompt:
        # Clear previous XAI chart on new prompt
        if "last_shap_chart" in st.session_state:
            del st.session_state["last_shap_chart"]
            
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                if current_provider == "Gemini (Google)":
                    # Initialize Gemini Model with programmatically selected model name
                    selected_model_name = st.session_state.get("gemini_selected_model", "gemini-1.5-flash")
                    chat_model = genai.GenerativeModel(
                        model_name=selected_model_name,
                        system_instruction=system_instruction,
                        tools=[get_simulation_result, get_shap_analysis, run_dr_simulation, get_top_regions, generate_report_from_conversation]
                    )
                    
                    # Setup chat conversation with context history
                    formatted_history = []
                    for msg in st.session_state.messages[:-1]: # Exclude the current prompt
                        gemini_role = "user" if msg["role"] == "user" else "model"
                        formatted_history.append({
                            "role": gemini_role,
                            "parts": [msg["content"]]
                        })
                    
                    chat = chat_model.start_chat(history=formatted_history)
                    
                    # Fetch response
                    with st.spinner("AI 분석가가 데이터를 분석 중입니다..."):
                        response = chat.send_message(prompt)
                        
                        # Function Calling loop - handles multiple/chained tool calls
                        max_iterations = 8
                        iteration = 0
                        while iteration < max_iterations:
                            iteration += 1
                            # Check if this response contains any function calls
                            has_function_call = False
                            if response.candidates and response.candidates[0].content.parts:
                                parts = response.candidates[0].content.parts
                                for part in parts:
                                    if hasattr(part, "function_call") and part.function_call:
                                        has_function_call = True
                                        name = part.function_call.name
                                        args = part.function_call.args
                                        
                                        if name == "get_simulation_result":
                                            region = args.get("region", "")
                                            count = int(args.get("count", 10))
                                            power_type = args.get("power_type", "급속")
                                            sim_res = get_simulation_result(region, count, power_type)
                                        elif name == "get_shap_analysis":
                                            region = args.get("region", "")
                                            sim_res = get_shap_analysis(region)
                                        elif name == "run_dr_simulation":
                                            region = args.get("region", "")
                                            intervention_type = args.get("intervention_type", "V2G_Peak_Shaving")
                                            sim_res = run_dr_simulation(region, intervention_type)
                                        elif name == "get_top_regions":
                                            n = int(args.get("n", 10))
                                            metric = args.get("metric", "전력_부하지수")
                                            sim_res = get_top_regions(n, metric)
                                        elif name == "generate_report_from_conversation":
                                            region = args.get("region", "")
                                            sim_res = generate_report_from_conversation(region)
                                        else:
                                            sim_res = "알 수 없는 함수 호출입니다."
                                            
                                        # Feedback loop to Gemini LLM with JSON payload fallback
                                        try:
                                            func_response_part = genai.types.Part.from_function_response(
                                                name=name,
                                                response={"result": sim_res}
                                            )
                                        except Exception:
                                            func_response_part = {
                                                "function_response": {
                                                    "name": name,
                                                    "response": {"result": sim_res}
                                                }
                                            }
                                        
                                        response = chat.send_message(func_response_part)
                                        break  # One function call per iteration; re-check the new response
                            
                            if not has_function_call:
                                break  # No more function calls, exit loop
                                        
                        # Extract final text response
                        try:
                            full_response = response.text
                        except Exception:
                            full_response = "응답 처리 중 오류가 발생했습니다. 다시 시도해 주세요."
                else:
                    # OpenAI-compatible multi-provider logic (Groq)
                    openai_tools = [
                        {
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
                        {
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
                        {
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
                        {
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
                        {
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
                    ]

                    # Setup messages list for OpenAI format
                    openai_messages = [
                        {"role": "system", "content": system_instruction}
                    ]
                    # Append history
                    for msg in st.session_state.messages[:-1]:
                        openai_role = "user" if msg["role"] == "user" else "assistant"
                        openai_messages.append({
                            "role": openai_role,
                            "content": msg["content"]
                        })
                    # Add current user prompt
                    openai_messages.append({
                        "role": "user",
                        "content": prompt
                    })

                    with st.spinner("AI 분석가가 데이터를 분석 중입니다..."):
                        content, tool_calls = call_openai_compatible_api(
                            current_provider, active_key, openai_messages, openai_tools
                        )
                        
                        if tool_calls:
                            # Add assistant's message with tool call to context
                            openai_messages.append({
                                "role": "assistant",
                                "content": content,
                                "tool_calls": tool_calls
                            })
                            
                            # Process each tool call
                            for tool_call in tool_calls:
                                tc_id = tool_call.get("id")
                                func_info = tool_call.get("function")
                                name = func_info.get("name")
                                args = json.loads(func_info.get("arguments", "{}"))
                                
                                # Execute python function
                                if name == "get_simulation_result":
                                    region = args.get("region", "")
                                    count = int(args.get("count", 10))
                                    power_type = args.get("power_type", "급속")
                                    sim_res = get_simulation_result(region, count, power_type)
                                elif name == "get_shap_analysis":
                                    region = args.get("region", "")
                                    sim_res = get_shap_analysis(region)
                                elif name == "run_dr_simulation":
                                    region = args.get("region", "")
                                    intervention_type = args.get("intervention_type", "V2G_Peak_Shaving")
                                    sim_res = run_dr_simulation(region, intervention_type)
                                elif name == "get_top_regions":
                                    n = int(args.get("n", 10))
                                    metric = args.get("metric", "전력_부하지수")
                                    sim_res = get_top_regions(n, metric)
                                elif name == "generate_report_from_conversation":
                                    region = args.get("region", "")
                                    sim_res = generate_report_from_conversation(region)
                                else:
                                    sim_res = "알 수 없는 함수 호출입니다."
                                    
                                # Append tool response
                                openai_messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc_id,
                                    "name": name,
                                    "content": sim_res
                                })
                                
                            # Second call to get final text response
                            content, _ = call_openai_compatible_api(
                                current_provider, active_key, openai_messages
                            )
                            
                        full_response = content
                    
                message_placeholder.markdown(full_response)
                
                # Append assistant response with optional PDF metadata
                msg_data = {"role": "assistant", "content": full_response}
                if "pending_pdf_bytes" in st.session_state and "pending_pdf_region" in st.session_state:
                    msg_data["pdf_bytes"] = st.session_state.pop("pending_pdf_bytes")
                    msg_data["pdf_region"] = st.session_state.pop("pending_pdf_region")
                st.session_state.messages.append(msg_data)
                
                # Immediately render download button for any newly generated PDF
                if msg_data.get("pdf_bytes") and msg_data.get("pdf_region"):
                    st.download_button(
                        label=f"⬇️ {msg_data['pdf_region']} 관제 분석 보고서 다운로드 (PDF)",
                        data=msg_data["pdf_bytes"],
                        file_name=f"{msg_data['pdf_region']}_EV_charge_report.pdf",
                        mime="application/pdf",
                        key=f"dl_btn_new_{msg_data['pdf_region']}"
                    )
                
            except Exception as e:
                err_str = str(e)
                # API 키 등 보안에 민감한 정보가 에러 메시지에 포함되어 노출되지 않도록 마스킹 처리
                import re
                clean_err = re.sub(r'(Bearer\s+)[a-zA-Z0-9_\-\.\/]+', r'\1[MASKED_API_KEY]', err_str)
                clean_err = re.sub(r'(gsk_[a-zA-Z0-9_\-]+|sk\-[a-zA-Z0-9_\-]+)', '[MASKED_API_KEY]', clean_err)
                
                if "429" in clean_err or "quota" in clean_err.lower() or "ResourceExhausted" in clean_err:
                    st.error(
                        f"⚠️ **AI 서비스 API 호출 한도(Quota/Rate Limit)를 초과했습니다.**\n\n"
                        f"현재 사용 중인 {current_provider} API Key의 호출 제한에 도달했거나 할당량이 부족합니다. "
                        f"잠시 대기하신 후 다시 시도해 주세요.\n\n"
                        f"*상세 에러: {clean_err}*"
                    )
                else:
                    st.error(f"⚠️ **{current_provider} 호출 중 오류가 발생했습니다:** {clean_err}")
                
        # Rerun to update chat display. For PDF reports, generated_reports section at the top
        # will show the download button on the next render cycle, so rerun is safe here.
        st.rerun()

    # Render interactive Plotly SHAP chart if data exists in session state
    if st.session_state.get("last_shap_chart"):
        chart_info = st.session_state["last_shap_chart"]
        r_name = chart_info["region"]
        usage = chart_info["usage"]
        features = chart_info["features"]
        values = chart_info["values"]
        
        import plotly.express as px
        import pandas as pd
        
        df_shap = pd.DataFrame({
            "변수명": features,
            "기여도 (SHAP)": values,
            "영향": ["상승 기여" if v >= 0 else "감소 기여" for v in values]
        })
        
        df_shap["abs_val"] = df_shap["기여도 (SHAP)"].abs()
        df_shap = df_shap.sort_values("abs_val", ascending=True).tail(8) # Top 8
        
        fig = px.bar(
            df_shap,
            x="기여도 (SHAP)",
            y="변수명",
            color="영향",
            orientation="h",
            color_discrete_map={"상승 기여": "#FF5A5F", "감소 기여": "#3B82F6"},
            title=f"📊 {r_name} ({usage}) 예측 변수 영향도 (SHAP Top 8)"
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("❌ 차트 닫기", use_container_width=True):
            del st.session_state["last_shap_chart"]
            st.rerun()
