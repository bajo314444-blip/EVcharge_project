import warnings
import streamlit as st
import pandas as pd

from utils.data_processing import load_all_data, load_highway_data
from utils.models import train_models

# Import components
from components.sidebar import render_sidebar, render_urban_sidebar

# Import views
from views.urban_dashboard import render_dashboard, render_report
from views.highway_dashboard import render_highway_dashboard

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="수도권 전기차 충전소 부하 예측",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("수도권 전기차 충전소 부하 예측 웹서비스")
st.caption("자가용과 사업자용 수요를 비교하고, 신규 충전소 설치 시 부하 완화 효과를 지도에서 확인합니다.")

# 1. Sidebar Control
control_mode, data_dir = render_sidebar()

# 2. Main Logic
if control_mode == "도심 행정구역 관제":
    # State Management for Urban Data
    if "final_data" not in st.session_state or st.session_state.get("urban_data_dir") != data_dir:
        try:
            with st.spinner("데이터 로딩 및 모델 학습 중..."):
                final_data, monthly_data, hourly_data = load_all_data(data_dir)
                model_state = train_models(final_data.to_json(orient="split"), use_smote=False)
                model_state_smote = train_models(final_data.to_json(orient="split"), use_smote=True)
                
                st.session_state["final_data"] = final_data
                st.session_state["monthly_data"] = monthly_data
                st.session_state["hourly_data"] = hourly_data
                st.session_state["model_state"] = model_state
                st.session_state["model_state_smote"] = model_state_smote
                st.session_state["urban_data_dir"] = data_dir
        except Exception as exc:
            st.error(f"데이터를 불러오지 못했습니다: {exc}")
            st.stop()

    final_data = st.session_state["final_data"]
    monthly_data = st.session_state["monthly_data"]
    hourly_data = st.session_state["hourly_data"]
    model_state = st.session_state["model_state"]
    model_state_smote = st.session_state["model_state_smote"]

    usage_options, metric, province_filter = render_urban_sidebar(final_data)

    filtered = final_data[final_data["시도"].isin(province_filter)].copy()
    if usage_options:
        filtered = filtered[filtered["용도"].isin(usage_options)].copy()

    top_region = filtered.sort_values(metric, ascending=False).head(1)

    current_view = st.query_params.get("view", "dashboard")
    active_dash = "active" if current_view == "dashboard" else ""
    active_report = "active" if current_view == "report" else ""

    st.markdown(f'''
    <style>
    .bookmark-container {{
        position: fixed;
        right: 0;
        top: 25%;
        display: flex;
        flex-direction: column;
        gap: 15px;
        z-index: 999999;
    }}
    .bookmark-tab {{
        background-color: #FFFFFF;
        color: #1F2937 !important;
        writing-mode: vertical-rl;
        text-orientation: upright;
        padding: 20px 8px;
        border-radius: 12px 0 0 12px;
        text-decoration: none;
        font-weight: 800;
        font-size: 15px;
        letter-spacing: 2px;
        box-shadow: -4px 4px 15px rgba(0,0,0,0.08);
        transition: all 0.2s ease;
        border: 2px solid #E5E7EB;
        border-right: none;
    }}
    .bookmark-tab:hover {{
        transform: translateX(-5px);
        background-color: #F3F4F6;
    }}
    .bookmark-tab.active {{
        background-color: #00A699;
        color: #FFFFFF !important;
        border-color: #00A699;
        transform: translateX(-5px);
    }}
    </style>
    <div class="bookmark-container">
        <a href="?view=dashboard" class="bookmark-tab {active_dash}" target="_self">대시보드</a>
        <a href="?view=report" class="bookmark-tab {active_report}" target="_self">정책보고서</a>
    </div>
    ''', unsafe_allow_html=True)

    if current_view == "dashboard":
        render_dashboard(filtered, top_region, metric, usage_options, final_data, monthly_data, hourly_data, model_state, model_state_smote)
    elif current_view == "report":
        render_report(filtered, final_data, model_state)

else:
    # Highway mode
    if "hw_data" not in st.session_state or st.session_state.get("highway_data_dir") != data_dir:
        try:
            with st.spinner("고속도로 데이터 로딩 중..."):
                hw_data = load_highway_data(data_dir)
                st.session_state["hw_data"] = hw_data
                st.session_state["highway_data_dir"] = data_dir
        except Exception as exc:
            st.error(f"고속도로 데이터를 불러오지 못했습니다: {exc}")
            st.stop()
            
    hw_data = st.session_state["hw_data"]
    render_highway_dashboard(hw_data)
