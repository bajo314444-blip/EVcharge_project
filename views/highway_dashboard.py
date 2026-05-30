import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from streamlit_folium import st_folium
import networkx as nx

from utils.visualizations import render_highway_edge_plot
from utils.optimization import optimize_highway_chargers


def render_highway_dashboard(hw_data):
    # --- 고속도로망 최적화 관제 모드 ---
    if hw_data.empty:
        st.warning("고속도로 JSON 데이터를 찾을 수 없거나 파싱에 실패했습니다.")
        st.stop()
        
    with st.sidebar:
        st.header("교통 시나리오 시뮬레이터")
        traffic_scenario = st.selectbox(
            "교통 시나리오 모드 선택",
            ["평시 상태 (Normal)", "평일 출퇴근 혼잡 시간", "주말 장거리 이동 폭증", "명절 대이동 연휴 사태"]
        )
        ev_budget = st.slider("추가 확충 가능한 총 충전기 수 (대)", min_value=1, max_value=50, value=10)
        
    st.subheader("고속도로망 충전 인프라 확충 시뮬레이터")
    
    # 1. 트리거 발동 (시나리오별 부하 가중치 시뮬레이션)
    hw_sim = hw_data.copy()
    hw_sim["부하_예측점수"] = 50.0  # 초기 베이스라인 부하
    
    if traffic_scenario == "평일 출퇴근 혼잡 시간":
        hw_sim["부하_예측점수"] = hw_sim["부하_예측점수"] * 1.3
    elif traffic_scenario == "주말 장거리 이동 폭증":
        hw_sim["부하_예측점수"] = hw_sim["부하_예측점수"] * 1.5
    elif traffic_scenario == "명절 대이동 연휴 사태":
        hw_sim["부하_예측점수"] = hw_sim["부하_예측점수"] * 2.0
        
    hw_sim["부하_예측점수"] = np.clip(hw_sim["부하_예측점수"], 0, 100)
    
    # 2. LP 최적화 엔진 구동
    hw_optimized = optimize_highway_chargers(hw_sim, ev_budget)
    
    # 3. 레이아웃
    col_left, col_right = st.columns([1, 1.5])
    
    with col_left:
        st.markdown("#### 최적화 요약")
        st.metric("대상 노드(휴게소/IC) 수", f"{len(hw_optimized):,}개")
        st.metric("적용 시나리오", traffic_scenario)
        st.metric("최적 설치 제안 총합", f"{hw_optimized['최적_추가대수'].sum():,}대")
        
        st.markdown("#### 최적화 상세 결과")
        st.dataframe(
            hw_optimized[["unitName", "routeName", "Max_Capacity", "총용량_kW", "부하_예측점수", "최적_추가대수", "최적화후_부하점수"]].sort_values("최적_추가대수", ascending=False),
            height=400,
            use_container_width=True
        )
        
    with col_right:
        st.markdown("#### 상하행선 분리 네트워크 에지 맵")
        try:
            fig = render_highway_edge_plot(hw_optimized, traffic_scenario)
            fig.update_layout(height=700, margin=dict(l=0, r=0, t=30, b=0)) # 정사각형(크게)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"시각화 렌더링 오류: {e}")

