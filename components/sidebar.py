# ============================================================
# 파일명: sidebar.py
# 설명: Streamlit(스트림릿) 사이드바 컴포넌트 모듈.
#       관제 모드 선택, 데이터 폴더 설정, 지도 필터,
#       관제 옵션 등 사이드바 UI 요소들을 렌더링한다.
# ============================================================

import streamlit as st  # streamlit(스트림릿) 웹 앱 프레임워크를 st로 import(임포트)
from utils.data_processing import DEFAULT_DATA_DIR, METRO_SHORT  # 데이터 처리 모듈에서 기본 데이터 경로와 수도권 약칭 리스트를 import(임포트)

# --- 메인 사이드바(Sidebar) 렌더링 함수 ---
def render_sidebar():  # 사이드바에서 관제 모드와 데이터 디렉터리를 선택하는 함수 정의
    with st.sidebar:  # Streamlit(스트림릿) sidebar(사이드바) context(컨텍스트) 진입
        st.header("시스템 설정")  # 사이드바에 "시스템 설정" 헤더(제목) 표시
        control_mode = st.radio("관제 모드 선택", ["도심 행정구역 관제", "고속도로망 최적화"])  # radio(라디오) 버튼으로 관제 모드 선택 UI 렌더링
        st.header("데이터 설정")  # 사이드바에 "데이터 설정" 헤더(제목) 표시
        data_dir = st.text_input("CSV/Excel 데이터 폴더", value=str(DEFAULT_DATA_DIR))  # text_input(텍스트 입력)으로 데이터 폴더 경로 입력 UI 렌더링
        st.caption("폴더 안의 원본 파일명을 그대로 사용합니다.")  # 데이터 폴더 입력 하단에 안내 캡션 표시
        return control_mode, data_dir  # 선택된 관제 모드와 데이터 디렉터리 경로를 tuple(튜플)로 반환

# --- 도심 모드 전용 사이드바(Sidebar) 필터 렌더링 함수 ---
def render_urban_sidebar(final_data):  # 도심 관제 모드의 추가 필터 옵션을 렌더링하는 함수 정의 (final_data: 전체 데이터 DataFrame)
    with st.sidebar:  # Streamlit(스트림릿) sidebar(사이드바) context(컨텍스트) 진입
        st.header("지도 필터")  # 사이드바에 "지도 필터" 헤더(제목) 표시
        usage_options = st.multiselect("용도", ["자가용", "사업자용"], default=["자가용", "사업자용"])  # multiselect(다중선택)로 용도 필터 UI 렌더링 (기본: 전체 선택)

        # --- 3가지 변수를 각각 최댓값 대비 백분율(0~100)로 스케일링(Scaling)한 후 합산하여 '전체' 변수 생성 ---
        if "전체(3종 합산)" not in final_data.columns:  # DataFrame(데이터프레임)에 "전체(3종 합산)" 컬럼이 아직 없는 경우
            final_data["전체(3종 합산)"] = (final_data["전력_부하지수"] / final_data["전력_부하지수"].max()) * 100 + \
                                       (final_data["인프라_부하지수"] / final_data["인프라_부하지수"].max()) * 100 + \
                                       (final_data["총_전력판매량"] / final_data["총_전력판매량"].max()) * 100  # 3종 지표를 각각 최댓값 기준 백분율로 환산 후 합산하여 신규 컬럼 생성

        metric = st.selectbox("버블 크기 기준", ["전체(3종 합산)", "전력_부하지수", "인프라_부하지수", "총_전력판매량"])  # selectbox(선택상자)로 지도 버블 크기 기준 metric(지표) 선택 UI 렌더링
        province_filter = st.multiselect("시도", METRO_SHORT, default=METRO_SHORT)  # multiselect(다중선택)로 시도 필터 UI 렌더링 (기본: 수도권 전체 선택)

        st.header("관제 옵션")  # 사이드바에 "관제 옵션" 헤더(제목) 표시
        if "show_warning_banner" not in st.session_state:  # session_state(세션 상태)에 경보 배너 토글 값이 없는 경우
            st.session_state["show_warning_banner"] = True  # 기본값으로 경보 배너 활성화(True) 설정
        show_banner = st.toggle("🚨 실시간 이상 경보 배너 활성화", value=st.session_state["show_warning_banner"])  # toggle(토글) 스위치로 경보 배너 활성/비활성 UI 렌더링
        st.session_state["show_warning_banner"] = show_banner  # 사용자가 선택한 토글 값을 session_state(세션 상태)에 저장

        return usage_options, metric, province_filter  # 용도 옵션, 지표, 시도 필터를 tuple(튜플)로 반환
