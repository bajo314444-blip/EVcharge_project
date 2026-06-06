import streamlit as st
from utils.data_processing import DEFAULT_DATA_DIR, METRO_SHORT

def render_sidebar():
    with st.sidebar:
        st.header("시스템 설정")
        control_mode = st.radio("관제 모드 선택", ["도심 행정구역 관제", "고속도로망 최적화"])
        st.header("데이터 설정")
        data_dir = st.text_input("CSV/Excel 데이터 폴더", value=str(DEFAULT_DATA_DIR))
        st.caption("폴더 안의 원본 파일명을 그대로 사용합니다.")
        return control_mode, data_dir

def render_urban_sidebar(final_data):
    with st.sidebar:
        st.header("지도 필터")
        usage_options = st.multiselect("용도", ["자가용", "사업자용"], default=["자가용", "사업자용"])
        
        # 3가지 변수를 각각 최댓값 대비 백분율(0~100)로 스케일링한 후 합산하여 '전체' 변수 생성
        if "전체(3종 합산)" not in final_data.columns:
            final_data["전체(3종 합산)"] = (final_data["전력_부하지수"] / final_data["전력_부하지수"].max()) * 100 + \
                                       (final_data["인프라_부하지수"] / final_data["인프라_부하지수"].max()) * 100 + \
                                       (final_data["총_전력판매량"] / final_data["총_전력판매량"].max()) * 100
                                       
        metric = st.selectbox("버블 크기 기준", ["전체(3종 합산)", "전력_부하지수", "인프라_부하지수", "총_전력판매량"])
        province_filter = st.multiselect("시도", METRO_SHORT, default=METRO_SHORT)
        
        st.header("관제 옵션")
        st.toggle("🚨 실시간 이상 경보 배너 활성화", value=True, key="show_warning_banner")
        
        return usage_options, metric, province_filter
