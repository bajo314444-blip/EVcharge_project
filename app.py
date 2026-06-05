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
from views.ai_assistant import render_ai_assistant

warnings.filterwarnings("ignore")

# 1. Matplotlib 한글 폰트 설정 (OS 전체 폰트 스캔 방지 및 한글 깨짐 해결)
try:
    import matplotlib as mpl
    import matplotlib.font_manager as fm
    import os
    
    font_path = os.path.join(os.path.dirname(__file__), "utils", "fonts", "NanumGothic.ttf")
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        mpl.rc('font', family='Malgun Gothic')
        mpl.rcParams['axes.unicode_minus'] = False
except Exception as e:
    pass

# 2. Plotly 전역 폰트 Noto Sans KR로 강제 적용
try:
    import plotly.io as pio
    if "streamlit" in pio.templates:
        pio.templates["streamlit"].layout.font.family = "Noto Sans KR"
    pio.templates.default = "streamlit"
except Exception as e:
    pass

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
            with st.spinner("데이터 로딩 및 사전 학습 모델 복원 중..."):
                final_data, monthly_data, hourly_data = load_all_data(data_dir)
                
                import os
                import joblib
                model_path = os.path.join("results", "trained_model_state.joblib")
                
                model_state = None
                model_state_smote = None
                
                if os.path.exists(model_path):
                    try:
                        # 사전 학습된 모델 패키지 즉시 로드 (0.1초 소요)
                        package = joblib.load(model_path)
                        model_state = package["model_state"]
                        model_state_smote = package.get("model_state_smote", None)
                        st.session_state["model_state_smote"] = model_state_smote
                    except Exception as e:
                        st.warning(f"사전 학습된 모델 파일 로드에 실패하여 즉석 학습으로 전환합니다. (사유: {e})")
                        model_state = None
                
                if model_state is None:
                    # 폴백: 파일이 없거나 로드에 실패한 경우 실시간으로 즉석 학습 수행
                    model_state = train_models(final_data.to_json(orient="split"), use_smote=False)
                    if "model_state_smote" in st.session_state:
                        del st.session_state["model_state_smote"]
                
                st.session_state["final_data"] = final_data
                st.session_state["monthly_data"] = monthly_data
                st.session_state["hourly_data"] = hourly_data
                st.session_state["model_state"] = model_state
                st.session_state["urban_data_dir"] = data_dir
        except Exception as exc:
            st.error(f"데이터를 불러오지 못했습니다: {exc}")
            st.stop()

    final_data = st.session_state["final_data"]
    monthly_data = st.session_state["monthly_data"]
    hourly_data = st.session_state["hourly_data"]
    model_state = st.session_state["model_state"]
    model_state_smote = st.session_state.get("model_state_smote", None)

    usage_options, metric, province_filter = render_urban_sidebar(final_data)

    filtered = final_data[final_data["시도"].isin(province_filter)].copy()
    if usage_options:
        filtered = filtered[filtered["용도"].isin(usage_options)].copy()

    top_region = filtered.sort_values(metric, ascending=False).head(1)

    current_view = st.query_params.get("view", "dashboard")
    active_dash = "active" if current_view == "dashboard" else ""
    active_report = "active" if current_view == "report" else ""
    active_ai = "active" if current_view == "ai" else ""

    st.markdown(f'''
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&display=swap');
    
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .main {{
        font-family: 'Noto Sans KR', sans-serif !important;
    }}
    
    /* Plotly 한글 폰트 상속 강제 */
    .js-plotly-plot .plotly .xaxislayer-above, 
    .js-plotly-plot .plotly .yaxislayer-above, 
    .js-plotly-plot .plotly .gtitle, 
    .js-plotly-plot .plotly .xtick, 
    .js-plotly-plot .plotly .ytick {{
        font-family: 'Noto Sans KR', sans-serif !important;
    }}

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
        <a href="?view=ai" class="bookmark-tab {active_ai}" target="_self">AI 관제비서</a>
    </div>
    ''', unsafe_allow_html=True)

    if current_view == "dashboard":
        render_dashboard(filtered, top_region, metric, usage_options, final_data, monthly_data, hourly_data, model_state, model_state_smote)
    elif current_view == "report":
        render_report(filtered, final_data, model_state)
    elif current_view == "ai":
        render_ai_assistant(filtered, model_state, control_mode)

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
