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

    # API Client verification
    model = get_gemini_client()
    
    if not model:
        st.warning("⚠️ **Gemini API 키가 등록되지 않았습니다.**")
        st.info("Streamlit Secrets에 `GEMINI_API_KEY`를 등록하시거나, 아래 사이드바 하단에서 API 키를 임시 입력해 주세요.")
        
        # Render dynamic API Key input inside sidebar as a fallback
        with st.sidebar:
            st.markdown("---")
            st.subheader("🔑 API 설정 (Fallback)")
            user_key = st.text_input("Gemini API Key 입력", type="password", key="user_gemini_key_input")
            if user_key:
                st.session_state["user_gemini_api_key"] = user_key
                st.rerun()
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
                <h4>📋 보고서 요약 요청</h4>
                <p>현재 도심 행정구역 또는 고속도로망 최적화의 핵심 요약과 당면 문제점을 보고서용으로 정돈해 줍니다.</p>
            </div>
        """, unsafe_allow_html=True)

    # Context Injection setup
    dashboard_context = build_system_context(filtered_data, model_state, control_mode, hw_data)
    
    system_instruction = f"""
    당신은 '수도권 전기차 충전소 부하 예측 및 설치 시뮬레이션 서비스'의 전문 AI 관제 비서인 'EV-Charge AI'입니다.
    사용자의 질문에 친절하고 신뢰감 있는 정부 보고서 서기 스타일의 한국어로 대답하십시오.
    
    현재 대시보드의 실시간 통계 및 예측 상태 데이터(JSON 형식):
    {dashboard_context}
    
    답변 지침:
    1. 사용자가 특정 지역의 상태나 전체 데이터의 특징을 묻는 경우, 위의 실시간 요약 데이터(JSON)를 바탕으로 정확한 통계 수치를 인용하여 설명하십시오.
    2. 주요 위험 지역(TOP 3 고위험 우려지역 등)의 전력 및 인프라 부하지수를 해석해 주십시오.
    3. 예측 모델에 관해 묻는다면, 현재 최적 모델명과 Test RMSE 오차를 토대로 과학적 타당성을 설명해 주십시오.
    4. 분석 내용에 표나 불릿 포인트를 적극적으로 활용하여 공공 보고서처럼 가독성 높게 답변하십시오.
    """

    # Session State Chat History Init
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

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
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                # Initialize Gemini Model with programmatically selected model name
                selected_model_name = st.session_state.get("gemini_selected_model", "gemini-1.5-flash")
                chat_model = genai.GenerativeModel(
                    model_name=selected_model_name,
                    system_instruction=system_instruction,
                    tools=[get_simulation_result]
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
                    
                    # Function Calling loop to handle simulation requests
                    if response.candidates and response.candidates[0].content.parts:
                        parts = response.candidates[0].content.parts
                        for part in parts:
                            if hasattr(part, "function_call") and part.function_call:
                                name = part.function_call.name
                                args = part.function_call.args
                                
                                if name == "get_simulation_result":
                                    region = args.get("region", "")
                                    count = int(args.get("count", 10))
                                    power_type = args.get("power_type", "급속")
                                    
                                    # Execute precomputed check / real-time fallback
                                    sim_res = get_simulation_result(region, count, power_type)
                                    
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
                                    break
                                    
                    full_response = response.text
                    
                message_placeholder.markdown(full_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Gemini API 호출 중 오류가 발생했습니다: {e}")
                
        # Rerun to clear suggestion button trigger state cleanly
        if clicked_prompt:
            st.rerun()
