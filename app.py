# ============================================================
# 파일명: app.py
# 설명: 수도권 전기차 충전소 부하 예측 웹서비스의 메인 엔트리포인트.
#       Streamlit(스트림릿) 기반 대시보드를 구성하며,
#       도심 행정구역 관제와 고속도로망 최적화 두 가지 모드를 제공한다.
# ============================================================

import warnings  # warnings(경고) 모듈을 import(임포트)하여 경고 메시지 제어
import streamlit as st  # streamlit(스트림릿) 웹 앱 프레임워크를 st로 import(임포트)
import pandas as pd  # pandas(판다스) 데이터 분석 라이브러리를 pd로 import(임포트)

from utils.data_processing import load_all_data, load_highway_data  # 데이터 전처리 모듈에서 도심/고속도로 데이터 로드 함수를 import(임포트)
from utils.models import train_models  # 모델 유틸에서 학습 함수를 import(임포트)

# --- 컴포넌트(Component) import(임포트) ---
from components.sidebar import render_sidebar, render_urban_sidebar  # sidebar(사이드바) 컴포넌트에서 렌더링 함수들을 import(임포트)

# --- 뷰(View) import(임포트) ---
from views.urban_dashboard import render_dashboard, render_report  # 도심 대시보드 뷰에서 dashboard(대시보드)와 report(보고서) 렌더 함수 import(임포트)
from views.highway_dashboard import render_highway_dashboard  # 고속도로 대시보드 뷰에서 렌더 함수 import(임포트)
from views.ai_assistant import render_ai_assistant  # AI 관제비서 뷰에서 렌더 함수 import(임포트)

warnings.filterwarnings("ignore")  # 모든 warning(경고) 메시지를 무시하도록 필터 설정

# --- 1. Matplotlib(맷플롯립) 한글 폰트 설정 (OS 전체 폰트 스캔 방지 및 한글 깨짐 해결) ---
try:  # 폰트 설정 중 예외 발생 시 무시하기 위한 try 블록 시작
    import matplotlib as mpl  # matplotlib(맷플롯립) 시각화 라이브러리를 mpl로 import(임포트)
    import matplotlib.font_manager as fm  # font_manager(폰트 매니저) 모듈을 fm으로 import(임포트)
    import os  # os(운영체제) 인터페이스 모듈을 import(임포트)

    font_path = os.path.join(os.path.dirname(__file__), "utils", "fonts", "NanumGothic.ttf")  # 나눔고딕 폰트 파일의 절대 경로를 생성
    if os.path.exists(font_path):  # 해당 폰트 파일이 존재하는지 확인
        fm.fontManager.addfont(font_path)  # fontManager(폰트 매니저)에 나눔고딕 폰트를 등록
        mpl.rc('font', family='Malgun Gothic')  # matplotlib(맷플롯립) 기본 폰트를 맑은 고딕으로 설정
        mpl.rcParams['axes.unicode_minus'] = False  # 마이너스 기호가 유니코드로 깨지는 현상 방지
except Exception as e:  # 폰트 설정 중 발생하는 모든 예외를 포착
    pass  # 예외 발생 시 아무 작업도 수행하지 않고 무시

# --- 2. Plotly(플로틀리) 전역 폰트를 Noto Sans KR로 강제 적용 ---
try:  # Plotly(플로틀리) 폰트 설정 중 예외 발생 시 무시하기 위한 try 블록 시작
    import plotly.io as pio  # plotly.io 모듈을 pio로 import(임포트)하여 템플릿 설정
    if "streamlit" in pio.templates:  # pio.templates(템플릿 목록)에 "streamlit" 템플릿이 존재하는지 확인
        pio.templates["streamlit"].layout.font.family = "Noto Sans KR"  # streamlit 템플릿의 폰트를 Noto Sans KR로 변경
    pio.templates.default = "streamlit"  # 기본 template(템플릿)을 "streamlit"으로 설정
except Exception as e:  # Plotly(플로틀리) 설정 중 발생하는 모든 예외를 포착
    pass  # 예외 발생 시 아무 작업도 수행하지 않고 무시

# --- Streamlit(스트림릿) 페이지 기본 설정 ---
st.set_page_config(  # Streamlit(스트림릿) 페이지 전역 설정 함수 호출
    page_title="수도권 전기차 충전소 부하 예측",  # 브라우저 탭에 표시될 페이지 제목 설정
    layout="wide",  # 와이드(wide) 레이아웃으로 전체 화면 사용
    initial_sidebar_state="expanded",  # sidebar(사이드바)를 기본적으로 펼쳐서 표시
)

st.title("수도권 전기차 충전소 부하 예측 웹서비스")  # 메인 페이지 상단에 대제목 표시
st.caption("자가용과 사업자용 수요를 비교하고, 신규 충전소 설치 시 부하 완화 효과를 지도에서 확인합니다.")  # 대제목 하단에 부가 설명 캡션 표시

# --- 1. Sidebar(사이드바) 관제 모드 및 데이터 경로 선택 ---
control_mode, data_dir = render_sidebar()  # sidebar(사이드바)를 렌더링하고 관제 모드와 데이터 디렉터리 경로를 반환받음

# --- 2. 메인 로직: 관제 모드에 따라 분기 처리 ---
if control_mode == "도심 행정구역 관제":  # 관제 모드가 "도심 행정구역 관제"인 경우 분기

    # --- 도심 데이터 State(상태) 관리 및 로딩 ---
    if "final_data" not in st.session_state or st.session_state.get("urban_data_dir") != data_dir:  # session_state(세션 상태)에 데이터가 없거나 데이터 경로가 변경된 경우
        try:  # 데이터 로딩 중 예외 발생 시 처리하기 위한 try 블록 시작
            with st.spinner("데이터 로딩 및 사전 학습 모델 복원 중..."):  # 로딩 중 spinner(스피너) 애니메이션 표시
                final_data, monthly_data, hourly_data = load_all_data(data_dir)  # 지정 경로에서 최종/월별/시간별 데이터를 로드

                import os  # os(운영체제) 모듈을 import(임포트) (스코프 내 재사용)
                from utils.data_processing import load_precomputed_analytics  # 사전 계산 분석 결과 로드 함수 import(임포트)

                json_path = os.path.join("results", "precomputed_analytics.json")  # 사전 계산된 분석 결과 JSON 파일 경로 생성
                onnx_path = os.path.join("results", "best_model.onnx")  # 사전 학습된 ONNX 모델 파일 경로 생성

                model_state = None  # model_state(모델 상태) 변수를 None으로 초기화
                model_state_smote = None  # SMOTE(오버샘플링) 적용 모델 상태 변수를 None으로 초기화

                if os.path.exists(json_path) and os.path.exists(onnx_path):  # JSON과 ONNX 파일이 모두 존재하는지 확인
                    try:  # 모델 복원 중 예외 처리를 위한 try 블록 시작
                        # ONNX + JSON 이원화 아키텍처(Architecture)로 모델 즉시 복원 (약 0.01초 소요)
                        model_state, model_state_smote = load_precomputed_analytics(json_path, onnx_path)  # 사전 계산 파일에서 모델 상태 복원
                        st.session_state["model_state_smote"] = model_state_smote  # SMOTE 모델 상태를 session_state(세션 상태)에 저장
                    except Exception as e:  # 모델 복원 실패 시 예외 포착
                        st.warning(f"이원화 모델 파일 로드에 실패하여 즉석 학습으로 전환합니다. (사유: {e})")  # 사용자에게 경고 메시지 표시
                        model_state = None  # 모델 상태를 None으로 리셋하여 fallback(폴백) 학습 유도

                if model_state is None:  # 모델 상태가 여전히 None인 경우 (파일 미존재 또는 로드 실패)
                    # 폴백(Fallback): 파일이 없거나 로드에 실패한 경우 실시간으로 즉석 학습 수행
                    model_state = train_models(final_data.to_json(orient="split"), use_smote=False)  # 데이터를 JSON으로 변환하여 모델 학습 실행
                    if "model_state_smote" in st.session_state:  # session_state(세션 상태)에 SMOTE 모델이 남아있으면
                        del st.session_state["model_state_smote"]  # 이전 SMOTE 모델 상태를 삭제하여 정합성 유지

                st.session_state["final_data"] = final_data  # 최종 통합 데이터를 session_state(세션 상태)에 저장
                st.session_state["monthly_data"] = monthly_data  # 월별 데이터를 session_state(세션 상태)에 저장
                st.session_state["hourly_data"] = hourly_data  # 시간별 데이터를 session_state(세션 상태)에 저장
                st.session_state["model_state"] = model_state  # 학습된 모델 상태를 session_state(세션 상태)에 저장
                st.session_state["urban_data_dir"] = data_dir  # 현재 데이터 디렉터리 경로를 session_state(세션 상태)에 저장
        except Exception as exc:  # 데이터 로딩 전체 과정에서 발생하는 예외 포착
            st.error(f"데이터를 불러오지 못했습니다: {exc}")  # 사용자에게 에러 메시지 표시
            st.stop()  # Streamlit(스트림릿) 앱 실행을 즉시 중단

    # --- session_state(세션 상태)에서 캐싱된 데이터 로드 ---
    final_data = st.session_state["final_data"]  # session_state(세션 상태)에서 최종 데이터 가져오기
    monthly_data = st.session_state["monthly_data"]  # session_state(세션 상태)에서 월별 데이터 가져오기
    hourly_data = st.session_state["hourly_data"]  # session_state(세션 상태)에서 시간별 데이터 가져오기
    model_state = st.session_state["model_state"]  # session_state(세션 상태)에서 모델 상태 가져오기
    model_state_smote = st.session_state.get("model_state_smote", None)  # session_state(세션 상태)에서 SMOTE 모델 상태 가져오기 (없으면 None)

    usage_options, metric, province_filter = render_urban_sidebar(final_data)  # 도심 모드 전용 sidebar(사이드바) 필터 렌더링 및 선택값 반환

    # --- 데이터 필터링: 시도 및 용도 기준 ---
    filtered = final_data[final_data["시도"].isin(province_filter)].copy()  # 선택한 시도에 해당하는 행만 필터링하여 복사
    if usage_options:  # 용도 필터 옵션이 비어있지 않은 경우
        filtered = filtered[filtered["용도"].isin(usage_options)].copy()  # 선택한 용도에 해당하는 행만 추가 필터링하여 복사

    top_region = filtered.sort_values(metric, ascending=False).head(1)  # 선택한 metric(지표) 기준으로 내림차순 정렬 후 상위 1개 지역 추출

    # --- 현재 뷰(View) 모드 결정 (query_params 활용) ---
    current_view = st.query_params.get("view", "dashboard")  # URL query parameter(쿼리 파라미터)에서 현재 뷰 모드를 가져옴 (기본값: dashboard)
    active_dash = "active" if current_view == "dashboard" else ""  # dashboard(대시보드) 탭 활성화 CSS 클래스 결정
    active_report = "active" if current_view == "report" else ""  # report(보고서) 탭 활성화 CSS 클래스 결정
    active_ai = "active" if current_view == "ai" else ""  # AI 관제비서 탭 활성화 CSS 클래스 결정

    # --- 북마크(Bookmark) 탭 UI를 위한 HTML/CSS 마크업 삽입 ---
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
    ''', unsafe_allow_html=True)  # HTML 코드를 unsafe_allow_html(비보안 HTML 허용) 옵션으로 렌더링

    # --- 뷰(View) 모드에 따른 화면 렌더링 분기 ---
    if current_view == "dashboard":  # 현재 뷰가 "dashboard"인 경우
        render_dashboard(filtered, top_region, metric, usage_options, final_data, monthly_data, hourly_data, model_state, model_state_smote)  # 대시보드 뷰를 렌더링
    elif current_view == "report":  # 현재 뷰가 "report"인 경우
        render_report(filtered, final_data, model_state)  # 정책 보고서 뷰를 렌더링
    elif current_view == "ai":  # 현재 뷰가 "ai"인 경우
        render_ai_assistant(filtered, model_state, control_mode)  # AI 관제비서 뷰를 렌더링

# --- 고속도로망 최적화 관제 모드 ---
else:  # 관제 모드가 "고속도로망 최적화"인 경우 분기
    # --- 고속도로(Highway) 데이터 State(상태) 관리 및 로딩 ---
    if "hw_data" not in st.session_state or st.session_state.get("highway_data_dir") != data_dir:  # session_state(세션 상태)에 고속도로 데이터가 없거나 경로가 변경된 경우
        try:  # 고속도로 데이터 로딩 중 예외 처리를 위한 try 블록 시작
            with st.spinner("고속도로 데이터 로딩 중..."):  # 로딩 중 spinner(스피너) 애니메이션 표시
                hw_data = load_highway_data(data_dir)  # 지정 경로에서 고속도로 데이터를 로드
                st.session_state["hw_data"] = hw_data  # 고속도로 데이터를 session_state(세션 상태)에 저장
                st.session_state["highway_data_dir"] = data_dir  # 현재 고속도로 데이터 디렉터리 경로를 session_state(세션 상태)에 저장
        except Exception as exc:  # 고속도로 데이터 로딩 중 발생하는 예외 포착
            st.error(f"고속도로 데이터를 불러오지 못했습니다: {exc}")  # 사용자에게 에러 메시지 표시
            st.stop()  # Streamlit(스트림릿) 앱 실행을 즉시 중단

    hw_data = st.session_state["hw_data"]  # session_state(세션 상태)에서 고속도로 데이터 가져오기
    render_highway_dashboard(hw_data)  # 고속도로 대시보드 뷰를 렌더링
