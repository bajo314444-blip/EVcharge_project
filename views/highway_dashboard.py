# ============================================================
# 파일명: highway_dashboard.py
# 설명: 고속도로망 충전 인프라 확충 시뮬레이터 대시보드 뷰.
#       교통 시나리오별 부하 가중치를 시뮬레이션하고,
#       LP(선형계획법) 최적화 엔진을 구동하여 최적 충전기
#       배치 결과를 시각화한다.
# ============================================================

import streamlit as st  # streamlit(스트림릿) 웹 앱 프레임워크를 st로 import(임포트)
import pandas as pd  # pandas(판다스) 데이터 분석 라이브러리를 pd로 import(임포트)
import numpy as np  # numpy(넘파이) 수치 연산 라이브러리를 np로 import(임포트)
import plotly.express as px  # plotly.express(플로틀리 익스프레스) 간편 시각화 모듈을 px로 import(임포트)
import plotly.graph_objects as go  # plotly.graph_objects(그래프 오브젝트) 모듈을 go로 import(임포트)
from scipy import stats  # scipy.stats(사이파이 통계) 모듈을 import(임포트)
import matplotlib.pyplot as plt  # matplotlib.pyplot(맷플롯립 파이플롯) 시각화 모듈을 plt로 import(임포트)
import statsmodels.api as sm  # statsmodels(통계모델) API를 sm으로 import(임포트)
from sklearn.cross_decomposition import CCA  # sklearn(사이킷런)에서 CCA(정준상관분석) 클래스를 import(임포트)
from sklearn.decomposition import PCA  # sklearn(사이킷런)에서 PCA(주성분분석) 클래스를 import(임포트)
from sklearn.metrics import auc, roc_curve  # sklearn(사이킷런)에서 AUC, ROC curve(곡선) 함수를 import(임포트)
from sklearn.manifold import TSNE  # sklearn(사이킷런)에서 t-SNE(차원축소) 클래스를 import(임포트)
from sklearn.preprocessing import StandardScaler  # sklearn(사이킷런)에서 StandardScaler(표준화 스케일러) 클래스를 import(임포트)
from streamlit_folium import st_folium  # streamlit_folium(스트림릿 폴리움) 연동 모듈에서 st_folium 함수를 import(임포트)
import networkx as nx  # networkx(네트워크X) 그래프 분석 라이브러리를 nx로 import(임포트)

from utils.visualizations import render_highway_edge_plot  # 시각화 유틸에서 고속도로 에지(Edge) 플롯 렌더 함수를 import(임포트)
from utils.optimization import optimize_highway_chargers  # 최적화 유틸에서 고속도로 충전기 최적화 함수를 import(임포트)


# --- 고속도로 대시보드(Dashboard) 메인 렌더링 함수 ---
def render_highway_dashboard(hw_data):  # 고속도로 데이터를 받아 대시보드를 렌더링하는 함수 정의
    # --- 고속도로망 최적화 관제 모드 ---
    if hw_data.empty:  # 고속도로 데이터가 비어있는 경우 확인
        st.warning("고속도로 JSON 데이터를 찾을 수 없거나 파싱에 실패했습니다.")  # 사용자에게 경고 메시지 표시
        st.stop()  # Streamlit(스트림릿) 앱 실행을 즉시 중단

    # --- 사이드바(Sidebar): 교통 시나리오 시뮬레이터 설정 ---
    with st.sidebar:  # Streamlit(스트림릿) sidebar(사이드바) context(컨텍스트) 진입
        st.header("교통 시나리오 시뮬레이터")  # 사이드바에 "교통 시나리오 시뮬레이터" 헤더(제목) 표시
        traffic_scenario = st.selectbox(  # selectbox(선택상자)로 교통 시나리오 선택 UI 렌더링
            "교통 시나리오 모드 선택",  # selectbox(선택상자)의 레이블 텍스트
            ["평시 상태 (Normal)", "평일 출퇴근 혼잡 시간", "주말 장거리 이동 폭증", "명절 대이동 연휴 사태"]  # 선택 가능한 교통 시나리오 목록
        )
        ev_budget = st.slider("추가 확충 가능한 총 충전기 수 (대)", min_value=1, max_value=50, value=10)  # slider(슬라이더)로 추가 충전기 예산 수량 입력 UI 렌더링

    st.subheader("고속도로망 충전 인프라 확충 시뮬레이터")  # 메인 영역에 소제목(subheader) 표시

    # --- 1. 트리거(Trigger) 발동: 시나리오별 부하 가중치 시뮬레이션(Simulation) ---
    hw_sim = hw_data.copy()  # 원본 데이터를 변경하지 않기 위해 deep copy(깊은 복사) 생성
    hw_sim["부하_예측점수"] = 50.0  # 초기 베이스라인(Baseline) 부하 점수를 50.0으로 설정

    if traffic_scenario == "평일 출퇴근 혼잡 시간":  # 교통 시나리오가 "평일 출퇴근 혼잡 시간"인 경우
        hw_sim["부하_예측점수"] = hw_sim["부하_예측점수"] * 1.3  # 부하 점수에 1.3배 가중치를 적용
    elif traffic_scenario == "주말 장거리 이동 폭증":  # 교통 시나리오가 "주말 장거리 이동 폭증"인 경우
        hw_sim["부하_예측점수"] = hw_sim["부하_예측점수"] * 1.5  # 부하 점수에 1.5배 가중치를 적용
    elif traffic_scenario == "명절 대이동 연휴 사태":  # 교통 시나리오가 "명절 대이동 연휴 사태"인 경우
        hw_sim["부하_예측점수"] = hw_sim["부하_예측점수"] * 2.0  # 부하 점수에 2.0배 가중치를 적용

    hw_sim["부하_예측점수"] = np.clip(hw_sim["부하_예측점수"], 0, 100)  # 부하 점수를 0~100 범위로 clip(클리핑)하여 제한

    # --- 2. LP(선형계획법) 최적화 엔진(Engine) 구동 ---
    hw_optimized = optimize_highway_chargers(hw_sim, ev_budget)  # 시뮬레이션 데이터와 예산으로 최적 충전기 배치 계산

    # --- 3. 레이아웃(Layout) 구성: 좌측 요약 + 우측 시각화 ---
    col_left, col_right = st.columns([1, 1.5])  # 좌측(1):우측(1.5) 비율로 2개 column(컬럼) 레이아웃 생성

    with col_left:  # 좌측 column(컬럼) context(컨텍스트) 진입
        st.markdown("#### 최적화 요약")  # 좌측 영역에 "최적화 요약" 소제목 표시
        st.metric("대상 노드(휴게소/IC) 수", f"{len(hw_optimized):,}개")  # 최적화 대상 노드 수를 metric(메트릭) 카드로 표시
        st.metric("적용 시나리오", traffic_scenario)  # 현재 적용된 교통 시나리오를 metric(메트릭) 카드로 표시
        st.metric("최적 설치 제안 총합", f"{hw_optimized['최적_추가대수'].sum():,}대")  # 최적 추가 설치 충전기 총 대수를 metric(메트릭) 카드로 표시

        st.markdown("#### 최적화 상세 결과")  # "최적화 상세 결과" 소제목 표시
        st.dataframe(  # DataFrame(데이터프레임)을 인터랙티브 테이블로 표시
            hw_optimized[["unitName", "routeName", "Max_Capacity", "총용량_kW", "부하_예측점수", "최적_추가대수", "최적화후_부하점수"]].sort_values("최적_추가대수", ascending=False),  # 표시할 컬럼을 선택하고 최적 추가대수 기준 내림차순 정렬
            height=400,  # 테이블 높이를 400픽셀로 설정
            use_container_width=True  # 컨테이너 전체 너비를 사용하도록 설정
        )

    with col_right:  # 우측 column(컬럼) context(컨텍스트) 진입
        st.markdown("#### 상하행선 분리 네트워크 에지 맵")  # "상하행선 분리 네트워크 에지 맵" 소제목 표시
        try:  # 시각화 렌더링 중 예외 처리를 위한 try 블록 시작
            fig = render_highway_edge_plot(hw_optimized, traffic_scenario)  # 최적화 결과와 시나리오를 기반으로 에지(Edge) 플롯 Figure(피겨) 생성
            fig.update_layout(height=700, margin=dict(l=0, r=0, t=30, b=0))  # Figure(피겨) 레이아웃을 높이 700px, 여백 최소화로 업데이트
            st.plotly_chart(fig, use_container_width=True)  # Plotly(플로틀리) 차트를 Streamlit(스트림릿)에 렌더링
        except Exception as e:  # 시각화 렌더링 중 발생하는 모든 예외 포착
            st.error(f"시각화 렌더링 오류: {e}")  # 사용자에게 에러 메시지 표시
