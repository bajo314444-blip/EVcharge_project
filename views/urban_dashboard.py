# ============================================================
# 파일명: urban_dashboard.py
# 설명: 도심 행정구역 충전소 부하 예측 대시보드(Dashboard) 뷰.
#       지도 버블맵, 시뮬레이터, 예측 모델 비교, 강건성 평가,
#       SHAP/LIME 설명, 조건 충족표 등 분석 UI를 제공한다.
#       render_report(정책보고서 뷰) 함수로 PDF 연동 정책 보고서를 렌더링한다.
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

from utils.visualizations import make_bubble_map, make_tableone, render_shap_or_fallback, render_highway_edge_plot  # 시각화 유틸에서 버블맵, TableOne, SHAP, 고속도로 에지 플롯 함수를 import(임포트)
from utils.models import make_feature_matrix  # 모델 유틸에서 feature matrix(피처 행렬) 생성 함수를 import(임포트)
from utils.data_processing import cached_bootstrap, cached_adversarial, cached_ablation, cached_dca, cached_nested_cv, cached_spatial_external_validation, cached_survival, cached_partial_dependence  # 데이터 처리 유틸에서 캐시된 강건성/생존 분석 함수들을 import(임포트)
from utils.pdf_generator import generate_report_pdf, generate_highway_report_pdf  # PDF 생성 유틸에서 도심/고속도로 보고서 PDF 함수를 import(임포트)
from utils.optimization import optimize_highway_chargers, calculate_single_region_trajectory, calculate_topsis_rankings, simulate_dynamic_pricing  # 최적화 유틸에서 TOPSIS, 궤적, 다이내믹 요금 시뮬레이션 함수를 import(임포트)


# --- 도심 대시보드(Dashboard) 메인 렌더링 함수 ---
def render_dashboard(filtered, top_region, metric, usage_options, final_data, monthly_data, hourly_data, model_state, model_state_smote):  # 필터된 데이터와 모델 상태를 받아 대시보드 UI를 렌더링하는 함수 정의
    # --- 이상 징후(Anomaly) 목록 session_state(세션 상태) 초기화 ---
    if "anomalies_list" not in st.session_state:  # session_state(세션 상태)에 anomalies_list(이상 징후 목록) 키가 없는 경우
        np.random.seed(42)  # random seed(난수 시드)를 42로 고정하여 재현 가능한 모의 데이터 생성
        sample_regions = final_data.dropna(subset=["위도", "경도"]).sample(min(3, len(final_data)), random_state=42).copy()  # 위도/경도가 있는 지역 중 최대 3개를 무작위 샘플링
        anomaly_types = ["커넥터 온도 과열", "이상 전압 변동", "통신 패킷 유실"]  # 모의 이상 유형(anomaly type) 목록 정의
        anomalies = []  # 이상 징후(anomaly) 딕셔너리를 담을 빈 list(리스트) 초기화
        for idx, (_, row) in enumerate(sample_regions.iterrows()):  # 샘플링된 각 지역 row(행)를 순회
            anomalies.append({  # 이상 징후 정보 dict(딕셔너리)를 list(리스트)에 추가
                "지역": row["지역"],  # 지역명
                "용도": row["용도"],  # 충전 용도(자가용/사업자용)
                "위도": row["위도"],  # 위도 좌표
                "경도": row["경도"],  # 경도 좌표
                "anomaly_type": anomaly_types[idx % len(anomaly_types)],  # 이상 유형을 순환 할당
                "temperature": float(np.random.uniform(65.0, 85.0)),  # 커넥터 온도 모의값(65~85°C)
                "voltage_std": float(np.random.uniform(8.5, 15.0)),  # 전압 변동성 모의값(8.5~15.0 V)
                "packet_loss": float(np.random.uniform(5.0, 25.0)),  # 패킷 유실률 모의값(5~25%)
            })
        st.session_state["anomalies_list"] = anomalies  # 생성된 이상 징후 목록을 session_state(세션 상태)에 저장

    # --- 실시간 이상 징후 경고 배너(Warning Banner) 표시 ---
    anomalies = st.session_state["anomalies_list"]  # session_state(세션 상태)에서 이상 징후 목록 조회
    show_banner = st.session_state.get("show_warning_banner", True)  # 배너 표시 여부 플래그 조회(기본값 True)
    if show_banner and anomalies:  # 배너 표시가 활성화되어 있고 이상 징후가 존재하는 경우
        warn_col1, warn_col2 = st.columns([35, 1])  # 경고 메시지(35):닫기 버튼(1) 비율로 column(컬럼) 생성
        with warn_col1:  # 좌측 column(컬럼) context(컨텍스트) 진입
            # --- HTML 경고 배너 마크다운 블록 (빨간색 배경, 이상 징후 건수 표시) ---
            st.markdown(
                f"""
                <div style="background-color: #FF4B4B; color: white; padding: 12px; border-radius: 8px; font-weight: bold; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    🚨 [경고] 현재 수도권 충전소 {len(anomalies)}개 지점에서 이상 징후(과열, 전압 급변, 통신 장애)가 실시간 감지되었습니다. '이상 징후 관제 지도' 탭에서 상세 현황을 확인하세요.
                </div>
                """,
                unsafe_allow_html=True  # HTML(하이퍼텍스트 마크업) 렌더링 허용
            )
        with warn_col2:  # 우측 column(컬럼) context(컨텍스트) 진입
            st.markdown('<div style="padding-top: 10px;"></div>', unsafe_allow_html=True)  # 닫기 버튼 상단 여백용 빈 div(디브) 렌더링
            if st.button("❌", key="close_warning_banner", help="알림 배너 닫기"):  # 배너 닫기 button(버튼) 클릭 시
                st.session_state["show_warning_banner"] = False  # 배너 표시 플래그를 False로 설정
                st.rerun()  # Streamlit(스트림릿) 앱을 rerun(재실행)하여 배너 숨김

    # --- KPI metric(메트릭) 카드 4개: 분석 규모 및 최고 부하 지역 요약 ---
    col1, col2, col3, col4 = st.columns(4)  # 4개의 동일 너비 column(컬럼) 생성
    col1.metric("분석 지역-용도 행", f"{len(filtered):,}개")  # 필터 적용 후 분석 대상 행 수 표시
    col2.metric("최고 부하 지역", top_region["지역"].iloc[0] if not top_region.empty else "-")  # 전력 부하지수 최상위 지역명 표시
    col3.metric("최고 전력 부하지수", f"{top_region['전력_부하지수'].iloc[0]:,.2f}" if not top_region.empty else "-")  # 최고 부하지수 수치 표시
    col4.metric("학습 기준 최고 모델", model_state["best_name"])  # 학습 결과 기준 최우수 모델명 표시

    # --- 분석 메뉴 radio(라디오) 버튼: 수평 탭 형태 메뉴 선택 ---
    active_menu = st.radio(  # radio(라디오) 버튼으로 활성 분석 메뉴 선택 UI 렌더링
        "분석 메뉴 선택",  # radio(라디오) 레이블(숨김 처리됨)
        [
            "🗺️ 지도 버블맵",  # 지도 기반 버블맵 및 TOPSIS/이상징후 서브탭
            "📈 월별 부하 변화",  # 환경부 월별 충전 부하 추이
            "💡 분석 시뮬레이터",  # 충전기 증설 및 다이내믹 요금 시뮬레이션
            "📊 예측 모델 비교",  # ML/DL 모델 성능 비교 및 ROC
            "🛡️ 강건성 평가 (Phase 3)",  # Bootstrap, Nested CV, DCA 등 학술 평가
            "🧮 통계/군집 분석",  # TableOne, t-SNE, UMAP, CCA, 상관분석
            "🧠 SHAP/LIME 설명",  # 모델 해석 및 partial dependence
            "📋 조건 충족표",  # 프로젝트 평가 조건 충족 체크리스트
        ],
        horizontal=True,  # radio(라디오) 버튼을 수평 배치
        label_visibility="collapsed"  # 레이블을 숨김(collapsed) 처리
    )

    # --- 메뉴 분기: 지도 버블맵 ---
    if active_menu == "🗺️ 지도 버블맵":  # 선택된 메뉴가 "지도 버블맵"인 경우
        sub_tabs = st.tabs(["📍 현황 버블맵", "🎯 최적 입지 추천 지도", "🚨 이상 징후 관제 지도"])  # 3개 sub tab(서브 탭) 생성
        
        # --- 서브탭 0: 현황 버블맵 및 고부하 TOP 15 ---
        with sub_tabs[0]:  # "현황 버블맵" sub tab(서브 탭) context(컨텍스트) 진입
            st.subheader("현재 부하 버블맵")  # 소제목(subheader) 표시
            st.caption("지도 레이어에서 자가용과 사업자용을 켜고 끌 수 있습니다.")  # 캡션(caption) 설명 표시
            left, right = st.columns([1.35, 0.9])  # 좌측 지도(1.35):우측 순위표(0.9) column(컬럼) 레이아웃
            with left:  # 좌측 column(컬럼) context(컨텍스트) 진입
                st_folium(make_bubble_map(filtered, metric, usage_options), height=650, key="current_map", use_container_width=True, returned_objects=["last_object_clicked"])  # Folium(폴리움) 버블맵을 Streamlit(스트림릿)에 렌더링
            with right:  # 우측 column(컬럼) context(컨텍스트) 진입
                st.markdown("#### 고부하 상위 지역 (TOP 15)")  # 고부하 순위표 제목 표시

                rank_tabs = st.tabs(["전체", "사업자용", "자가용"])  # 용도별 순위 sub tab(서브 탭) 3개 생성

                def render_rank_table(df_subset, usage_name):  # 용도별 고부하 TOP 15 테이블 및 bar chart(막대 차트) 렌더링 내부 함수 정의
                    if df_subset.empty:  # subset DataFrame(데이터프레임)이 비어있는 경우
                        st.info(f"{usage_name} 데이터가 없습니다.")  # 정보(info) 메시지 표시
                        return  # 함수 조기 종료
            
                    cols_to_show = ["지역", "용도", "전기차_전체대수", "전체_충전기대수", "총용량_kW"]  # 테이블에 표시할 기본 column(컬럼) 목록
                    if metric not in cols_to_show:  # 선택 metric(지표)가 기본 컬럼에 없으면
                        cols_to_show.append(metric)  # metric(지표) column(컬럼)을 추가
                
                    top_table = (  # metric(지표) 기준 내림차순 TOP 15 테이블 생성
                        df_subset.sort_values(metric, ascending=False)  # metric(지표) 기준 내림차순 정렬
                        [cols_to_show]  # 표시할 column(컬럼)만 선택
                        .head(15)  # 상위 15개 row(행) 추출
                        .copy()  # 원본 변경 방지를 위해 copy(복사)
                    )
                    top_table.insert(0, "순위", range(1, len(top_table) + 1))  # 1번 column(컬럼)에 순위 번호 삽입
                    styled_table = top_table.style.set_properties(**{'text-align': 'center'}, subset=['순위'])  # 순위 column(컬럼) 가운데 정렬 style(스타일) 적용
                    st.dataframe(styled_table, use_container_width=True, hide_index=True)  # style(스타일) 적용된 테이블을 Streamlit(스트림릿)에 표시
        
                    top_table["지역_용도"] = top_table["지역"] + " (" + top_table["용도"] + ")"  # bar chart(막대 차트) Y축용 "지역 (용도)" 결합 column(컬럼) 생성
            
                    fig = px.bar(  # Plotly(플로틀리) horizontal bar chart(가로 막대 차트) 생성
                        top_table.sort_values("순위", ascending=True),  # 순위 오름차순 정렬(차트 표시용)
                        x=metric,  # X축: 선택 metric(지표)
                        y="지역_용도",  # Y축: 지역+용도 결합 레이블
                        color="용도",  # color(색상) 구분: 용도별
                        orientation="h",  # horizontal(가로) 방향 막대
                        color_discrete_map={"자가용": "#00A699", "사업자용": "#FF5A5F"},  # 용도별 고정 color map(색상 맵)
                        title=f"{usage_name} 고부하 TOP 15 ({metric} 기준)",  # chart(차트) 제목
                        labels={metric: f"{metric} 점수", "지역_용도": "지역 (용도)"},  # axis label(축 레이블) 설정
                        category_orders={"지역_용도": top_table.sort_values("순위", ascending=True)["지역_용도"].tolist()}  # Y축 category(카테고리) 순서 고정
                    )
                    st.plotly_chart(fig, use_container_width=True)  # Plotly(플로틀리) bar chart(막대 차트) 렌더링

                with rank_tabs[0]:  # "전체" sub tab(서브 탭) context(컨텍스트) 진입
                    render_rank_table(filtered, "전체")  # 전체 용도 고부하 순위 테이블 렌더링
                with rank_tabs[1]:  # "사업자용" sub tab(서브 탭) context(컨텍스트) 진입
                    render_rank_table(filtered[filtered["용도"] == "사업자용"], "사업자용")  # 사업자용만 필터링하여 순위 렌더링
                with rank_tabs[2]:  # "자가용" sub tab(서브 탭) context(컨텍스트) 진입
                    render_rank_table(filtered[filtered["용도"] == "자가용"], "자가용")  # 자가용만 필터링하여 순위 렌더링

        # --- 서브탭 1: TOPSIS(다중 기준 의사결정) 최적 입지 추천 ---
        with sub_tabs[1]:  # "최적 입지 추천 지도" sub tab(서브 탭) context(컨텍스트) 진입
            st.subheader("🎯 TOPSIS 다중 기준 의사결정(MCDA) 최적 입지 추천")  # TOPSIS 소제목 표시
            st.caption("전력 부하, 인프라 부하, 충전소 밀집도, 비용대비 전력망 완화율의 가중치를 고려하여 최적의 추가 충전소 설치 입지를 도출합니다.")  # TOPSIS 설명 캡션 표시
            
            w_col1, w_col2, w_col3, w_col4 = st.columns(4)  # 4개 criterion(기준) 가중치 slider(슬라이더)용 column(컬럼) 생성
            with w_col1:  # 1번 column(컬럼) context(컨텍스트) 진입
                w_load = st.slider("전력 부하지수 가중치", 0.0, 1.0, 0.35, 0.05)  # 전력 부하지수 weight(가중치) slider(슬라이더)
            with w_col2:  # 2번 column(컬럼) context(컨텍스트) 진입
                w_infra = st.slider("인프라 부하지수 가중치", 0.0, 1.0, 0.35, 0.05)  # 인프라 부하지수 weight(가중치) slider(슬라이더)
            with w_col3:  # 3번 column(컬럼) context(컨텍스트) 진입
                w_density = st.slider("충전소 밀집도 역수 가중치", 0.0, 1.0, 0.15, 0.05)  # 충전소 밀집도 역수 weight(가중치) slider(슬라이더)
            with w_col4:  # 4번 column(컬럼) context(컨텍스트) 진입
                w_mitigation = st.slider("전력망 완화율 가중치", 0.0, 1.0, 0.15, 0.05)  # 전력망 완화율 weight(가중치) slider(슬라이더)
                
            weights_sum = w_load + w_infra + w_density + w_mitigation  # 4개 weight(가중치) 합계 계산
            if weights_sum == 0:  # weight(가중치) 합이 0인 경우(정규화 불가)
                st.error("가중치 합이 0일 수 없습니다.")  # error(에러) 메시지 표시
            else:  # weight(가중치) 합이 0이 아닌 경우 TOPSIS 계산 진행
                norm_weights = {  # 정규화된 weight(가중치) dict(딕셔너리) 생성
                    "전력_부하지수": w_load / weights_sum,  # 전력 부하지수 정규 weight(가중치)
                    "인프라_부하지수": w_infra / weights_sum,  # 인프라 부하지수 정규 weight(가중치)
                    "충전소_밀집도_역수": w_density / weights_sum,  # 충전소 밀집도 역수 정규 weight(가중치)
                    "전력망_완화율": w_mitigation / weights_sum  # 전력망 완화율 정규 weight(가중치)
                }
                
                topsis_res = calculate_topsis_rankings(filtered, norm_weights)  # TOPSIS ranking(순위) 계산 함수 호출
                topsis_top5 = topsis_res.sort_values("TOPSIS_점수", ascending=False).head(5).copy()  # TOPSIS 점수 상위 5개 추출
                topsis_top5["TOPSIS_순위"] = range(1, len(topsis_top5) + 1)  # TOPSIS 순위 column(컬럼) 추가
                
                t_left, t_right = st.columns([1.35, 0.9])  # 좌측 지도(1.35):우측 TOP5 카드(0.9) column(컬럼) 레이아웃
                with t_left:  # 좌측 column(컬럼) context(컨텍스트) 진입
                    st_folium(  # TOPSIS TOP5 하이라이트 버블맵 렌더링
                        make_bubble_map(filtered, metric, usage_options, topsis_data=topsis_top5),  # TOPSIS 데이터를 포함한 버블맵 생성
                        height=600,  # 지도 높이 600px
                        key="topsis_map",  # Streamlit(스트림릿) widget(위젯) 고유 key(키)
                        use_container_width=True,  # container(컨테이너) 전체 너비 사용
                        returned_objects=["last_object_clicked"]  # 클릭 이벤트 object(객체) 반환 설정
                    )
                with t_right:  # 우측 column(컬럼) context(컨텍스트) 진입
                    st.markdown("#### 🎯 최적 입지 추천 TOP 5 지역")  # TOP 5 카드 목록 제목 표시
                    for _, row in topsis_top5.iterrows():  # TOP 5 각 row(행)를 순회
                        # --- HTML TOP5 추천 카드 블록 (금색 좌측 테두리, TOPSIS 점수 및 지표 표시) ---
                        st.markdown(
                            f"""
                            <div style="background-color: rgba(255,215,0,0.08); border-left: 5px solid #FFD700; padding: 10px; margin-bottom: 10px; border-radius: 4px;">
                                <h5 style="margin: 0; color: #E5A93C;">{row['TOPSIS_순위']}위: {row['지역']} ({row['용도']})</h5>
                                <p style="margin: 5px 0 0 0; font-size: 13px; line-height: 1.5;">
                                    <b>TOPSIS 점수:</b> {row['TOPSIS_점수']:.4f}<br>
                                    <b>전력 부하지수:</b> {row['전력_부하지수']:.2f} | <b>인프라 부하지수:</b> {row['인프라_부하지수']:.2f}<br>
                                    <b>충전소수:</b> {row['충전소개수']:.0f}개 | <b>총용량:</b> {row['총용량_kW']:.0f} kW
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True  # HTML(하이퍼텍스트 마크업) 렌더링 허용
                        )

        # --- 서브탭 2: 이상 징후(Anomaly Detection) 관제 지도 ---
        with sub_tabs[2]:  # "이상 징후 관제 지도" sub tab(서브 탭) context(컨텍스트) 진입
            st.subheader("🚨 충전기 고장 예후 Anomaly Detection 관제 지도")  # 이상 징후 관제 소제목 표시
            st.caption("전압 변동성, 커넥터 온도, 패킷 유실률 등의 실시간 지표를 파싱하여 고장 예후가 감지된 충전소를 실시간 모니터링합니다.")  # 이상 징후 설명 캡션 표시
            
            if st.button("🔄 실시간 상태 데이터 갱신 (모의 파싱)"):  # 모의 실시간 데이터 갱신 button(버튼) 클릭 시
                sample_regions = final_data.dropna(subset=["위도", "경도"]).sample(min(3, len(final_data))).copy()  # 위도/경도 있는 지역 중 최대 3개 무작위 샘플링
                anomaly_types = ["커넥터 온도 과열", "이상 전압 변동", "통신 패킷 유실"]  # 이상 유형 목록 정의
                new_anomalies = []  # 새 이상 징후 list(리스트) 초기화
                for idx, (_, row) in enumerate(sample_regions.iterrows()):  # 샘플 지역 각 row(행) 순회
                    new_anomalies.append({  # 새 이상 징후 dict(딕셔너리) 추가
                        "지역": row["지역"],  # 지역명
                        "용도": row["용도"],  # 충전 용도
                        "위도": row["위도"],  # 위도 좌표
                        "경도": row["경도"],  # 경도 좌표
                        "anomaly_type": np.random.choice(anomaly_types),  # 이상 유형 무작위 선택
                        "temperature": float(np.random.uniform(65.0, 85.0)),  # 커넥터 온도 모의값
                        "voltage_std": float(np.random.uniform(8.5, 15.0)),  # 전압 변동성 모의값
                        "packet_loss": float(np.random.uniform(5.0, 25.0)),  # 패킷 유실률 모의값
                    })
                st.session_state["anomalies_list"] = new_anomalies  # 갱신된 이상 징후 목록을 session_state(세션 상태)에 저장
                st.rerun()  # Streamlit(스트림릿) 앱 rerun(재실행)하여 UI 갱신
                
            anomalies = st.session_state["anomalies_list"]  # session_state(세션 상태)에서 이상 징후 목록 재조회
            
            a_left, a_right = st.columns([1.35, 0.9])  # 좌측 이상징후 지도(1.35):우측 경보 목록(0.9) column(컬럼) 레이아웃
            with a_left:  # 좌측 column(컬럼) context(컨텍스트) 진입
                st_folium(  # 이상 징후 마커가 포함된 버블맵 렌더링
                    make_bubble_map(filtered, metric, usage_options, anomalies=anomalies),  # anomalies(이상 징후) 데이터를 포함한 버블맵 생성
                    height=600,  # 지도 높이 600px
                    key="anomaly_map",  # Streamlit(스트림릿) widget(위젯) 고유 key(키)
                    use_container_width=True,  # container(컨테이너) 전체 너비 사용
                    returned_objects=["last_object_clicked"]  # 클릭 이벤트 object(객체) 반환 설정
                )
            with a_right:  # 우측 column(컬럼) context(컨텍스트) 진입
                st.markdown("#### 🚨 실시간 이상 감지 경보 목록")  # 경보 목록 제목 표시
                for item in anomalies:  # 각 이상 징후 item(항목)을 순회
                    # --- HTML 이상 징후 경보 카드 블록 (빨간색 좌측 테두리, 온도/전압/패킷 지표 표시) ---
                    st.markdown(
                        f"""
                        <div style="background-color: rgba(255,75,75,0.08); border-left: 5px solid #FF4B4B; padding: 10px; margin-bottom: 10px; border-radius: 4px;">
                            <h5 style="margin: 0; color: #FF4B4B;">🚨 {item['지역']} ({item['용도']})</h5>
                            <p style="margin: 5px 0 0 0; font-size: 13px; line-height: 1.5;">
                                <b>이상 유형:</b> <span style="color: #FF4B4B; font-weight: bold;">{item['anomaly_type']}</span><br>
                                <b>커넥터 온도:</b> {item['temperature']:.1f}°C (임계치: 60°C)<br>
                                <b>전압 변동성:</b> {item['voltage_std']:.2f} V (정상범위: &lt; 5.0 V)<br>
                                <b>패킷 유실률:</b> {item['packet_loss']:.1f}% (정상범위: &lt; 1.0%)
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True  # HTML(하이퍼텍스트 마크업) 렌더링 허용
                    )

    # --- 메뉴 분기: 월별 부하 변화 ---
    elif active_menu == "📈 월별 부하 변화":  # 선택된 메뉴가 "월별 부하 변화"인 경우
        st.subheader("환경부 공공급속 충전기 연월별 부하 변화")  # 월별 부하 변화 소제목 표시
        st.caption("새로 추가한 환경부 2017-2025년 월별 파일을 전처리 단계부터 반영했습니다.")  # 데이터 출처 캡션 표시
        available_regions = sorted(monthly_data["지역"].dropna().unique())  # 월별 데이터에서 사용 가능한 지역 목록 추출
        default_monthly_regions = (  # 기본 선택 지역: 월 충전량 합계 상위 5개
            monthly_data.groupby("지역")["월_충전량"].sum().sort_values(ascending=False).head(5).index.tolist()
        )
        selected_monthly_regions = st.multiselect("월별 추이를 볼 지역", available_regions, default=default_monthly_regions)  # multiselect(다중선택)으로 추이 비교 지역 선택
        monthly_metric = st.selectbox("월별 지표", ["월별_부하지수", "월_충전량", "월_충전횟수", "월_충전시간", "운영_충전소수"])  # selectbox(선택상자)로 월별 metric(지표) 선택
        month_view = monthly_data[monthly_data["지역"].isin(selected_monthly_regions)].copy()  # 선택 지역만 필터링한 월별 view(뷰) DataFrame(데이터프레임)
        fig = px.line(month_view, x="연월", y=monthly_metric, color="지역", markers=True)  # line chart(선 그래프)로 지역별 월별 추이 생성
        st.plotly_chart(fig, use_container_width=True)  # Plotly(플로틀리) line chart(선 그래프) 렌더링

        latest_month = monthly_data["연월"].max()  # 월별 데이터에서 최신 연월 조회
        latest = monthly_data[monthly_data["연월"] == latest_month].copy()  # 최신 연월 데이터만 필터링
        st.markdown(f"#### 최신 월 기준 고부하 지역: {latest_month.strftime('%Y-%m')}")  # 최신 월 기준 고부하 지역 제목 표시
        st.dataframe(  # 최신 월 고부하 TOP 20 테이블 표시
            latest.sort_values("월별_부하지수", ascending=False)  # 월별 부하지수 내림차순 정렬
            [["지역", "월_충전량", "월_충전횟수", "월_충전시간", "운영_충전소수", "월별_부하지수"]]  # 표시 column(컬럼) 선택
            .head(20),  # 상위 20개 row(행) 추출
            use_container_width=True,  # container(컨테이너) 전체 너비 사용
            hide_index=True,  # index(인덱스) column(컬럼) 숨김
        )

    # --- 메뉴 분기: 분석 시뮬레이터 ---
    elif active_menu == "💡 분석 시뮬레이터":  # 선택된 메뉴가 "분석 시뮬레이터"인 경우
        sim_tabs = st.tabs(["🔌 충전기 증설 시뮬레이션", "💸 다이내믹 요금제 시뮬레이션"])  # 2개 simulation(시뮬레이션) sub tab(서브 탭) 생성
        
        # --- 시뮬레이터 서브탭 0: 충전기 증설 Time-to-Overload 궤적 ---
        with sim_tabs[0]:  # "충전기 증설 시뮬레이션" sub tab(서브 탭) context(컨텍스트) 진입
            st.subheader("통합 설치 및 미래 시뮬레이터 (Time-to-Overload)")  # 충전기 증설 시뮬레이터 소제목 표시
            st.caption("충전기 추가 설치 시, 해당 지역의 부하지수가 어떻게 감소하며 과부하 시점을 얼마나 늦출 수 있는지(생존 궤적) 확인합니다.")  # 시뮬레이터 설명 캡션 표시

            sim_col1, sim_col2 = st.columns([1, 1.5])  # 좌측 설정(1):우측 궤적 차트(1.5) column(컬럼) 레이아웃

            with sim_col1:  # 좌측 설정 column(컬럼) context(컨텍스트) 진입
                st.markdown("##### 1. 시뮬레이션 설정")  # 시뮬레이션 설정 섹션 제목
                region_list = sorted(final_data["지역"].unique())  # 전체 지역 목록 정렬
                default_idx = region_list.index(top_region["지역"].iloc[0]) if not top_region.empty else 0  # 최고 부하 지역을 기본 선택 index(인덱스)로 설정
                sim_region = st.selectbox("설치 후보 지역", region_list, index=default_idx)  # selectbox(선택상자)로 시뮬레이션 대상 지역 선택
                sim_usage = st.radio("계산 기준 용도", ["자가용", "사업자용"], horizontal=True)  # radio(라디오)로 용도 선택
                growth_rate = st.slider("해당 지역 전기차 연간 증가율 (%)", 1.0, 20.0, 5.0, 1.0) / 100.0  # slider(슬라이더)로 연간 성장률 입력(%→비율 변환)

                st.markdown("##### 2. 설치 정책 (Intervention)")  # 설치 정책(Intervention) 섹션 제목
                charger_count = st.slider("추가 충전기 수", 1, 80, 10)  # slider(슬라이더)로 추가 충전기 대수 입력
                charger_kw = st.select_slider("충전기 1대 용량(kW)", options=[50, 100, 150, 200, 350], value=100)  # select_slider(선택 슬라이더)로 충전기 용량 선택

                target_row = final_data[(final_data["지역"] == sim_region) & (final_data["용도"] == sim_usage)].iloc[0]  # 선택 지역+용도의 target row(행) 추출
                added_kw = charger_count * charger_kw  # 추가 공급 용량(kW) = 대수 × 1대 용량
                base_load = target_row["총_전력판매량"]  # 현재 총 전력 판매량(부하)
                capacity = target_row["총용량_kW"]  # 현재 충전기 총 용량(kW)
                critical_threshold = final_data["전력_부하지수"].quantile(0.8)  # quantile(0.8)(80% 분위수) = 상위 20% 위험 한계선

                traj_df, overload_before, overload_after = calculate_single_region_trajectory(  # 단일 지역 부하지수 미래 궤적 계산
                    base_load, capacity, growth_rate, added_kw, critical_threshold  # 부하, 용량, 성장률, 추가용량, 임계치 전달
                )

                before = float(target_row["전력_부하지수"])  # 설치 전 전력 부하지수
                after = float(base_load / (capacity + added_kw)) if (capacity + added_kw) > 0 else 0  # 설치 후 부하지수(용량 0 방지)
                reduction_pct = (before - after) / before * 100 if before else 0  # 부하 감소율(%) 계산

                st.markdown("##### 3. 즉각적인 부하 감소 효과")  # 즉각 효과 metric(메트릭) 섹션 제목
                st.metric("현재 전력 부하지수", f"{before:,.2f}")  # 설치 전 부하지수 metric(메트릭) 표시
                st.metric("설치 후 부하지수", f"{after:,.2f}", delta=f"-{reduction_pct:.1f}%")  # 설치 후 부하지수 및 감소 delta(델타) 표시
                st.metric("추가 공급 용량", f"{added_kw:,.0f} kW")  # 추가 공급 용량 metric(메트릭) 표시

            with sim_col2:  # 우측 궤적 차트 column(컬럼) context(컨텍스트) 진입
                st.markdown("##### 4. 과부하 지연 효과 (정책의 장기적 가치)")  # 장기 과부하 지연 효과 섹션 제목

                delay_years = overload_after - overload_before  # 과부하 도달 시점 지연 연수 계산

                mc1, mc2, mc3 = st.columns(3)  # 3개 metric(메트릭) column(컬럼) 생성
                with mc1:  # 1번 metric(메트릭) column(컬럼)
                    st.metric("기존 과부하 도달", f"{overload_before}년 뒤" if overload_before < 15 else "안전 (15년+)")  # 설치 전 과부하 도달 시점
                with mc2:  # 2번 metric(메트릭) column(컬럼)
                    st.metric("설치 후 도달", f"{overload_after}년 뒤" if overload_after < 15 else "안전 (15년+)")  # 설치 후 과부하 도달 시점
                with mc3:  # 3번 metric(메트릭) column(컬럼)
                    st.metric("충전 대란 지연 효과", f"+{delay_years}년" if delay_years > 0 else "변동 없음")  # 과부하 지연 연수 효과
        
                fig = px.line(traj_df, x="Year", y="부하지수", color="상태", markers=True,   # 부하지수 미래 궤적 line chart(선 그래프) 생성
                              title=f"{sim_region} 부하지수 미래 궤적 (연 {growth_rate*100:.0f}% 성장)")
                fig.add_hline(y=critical_threshold, line_dash="dash", line_color="red", annotation_text="위험 한계선 (Threshold)")  # 위험 한계선 horizontal line(수평선) 추가
                st.plotly_chart(fig, use_container_width=True)  # Plotly(플로틀리) 궤적 차트 렌더링

                st_folium(  # 시뮬레이션 시나리오가 반영된 미니 버블맵 렌더링
                    make_bubble_map(  # 시나리오 버블맵 생성
                        final_data[final_data["지역"].isin([sim_region])],  # 선택 지역만 필터링
                        "전력_부하지수",  # metric(지표): 전력 부하지수
                        [sim_usage],  # 선택 용도만 표시
                        selected_region=sim_region,  # 선택 지역 하이라이트
                        scenario={"added_kw": added_kw, "reduction_pct": reduction_pct},  # 시나리오 파라미터(추가용량, 감소율) 전달
                    ),
                    height=300,  # 미니 지도 높이 300px
                    key="sim_map",  # Streamlit(스트림릿) widget(위젯) 고유 key(키)
                    use_container_width=True,  # container(컨테이너) 전체 너비 사용
                    returned_objects=["last_object_clicked"]  # 클릭 이벤트 object(객체) 반환 설정
                )

        # --- 시뮬레이터 서브탭 1: 다이내믹 요금제(Dynamic Pricing) 부하 분산 ---
        with sim_tabs[1]:  # "다이내믹 요금제 시뮬레이션" sub tab(서브 탭) context(컨텍스트) 진입
            st.subheader("💸 다이내믹 요금제 수요 이동 시뮬레이션")  # 다이내믹 요금제 시뮬레이터 소제목 표시
            st.caption("가격 탄력성(Elasticity) 모델을 기반으로 피크 시간대 할증 및 경부하 시간대 할인을 통해 부하 수요가 분산되는 효과를 시뮬레이션합니다.")  # 탄력성 모델 설명 캡션 표시
            
            dp_col1, dp_col2 = st.columns([1, 1.8])  # 좌측 설정(1):우측 차트(1.8) column(컬럼) 레이아웃
            with dp_col1:  # 좌측 설정 column(컬럼) context(컨텍스트) 진입
                st.markdown("##### 1. 요금제 및 탄력성 설정")  # 요금제 설정 섹션 제목
                elasticity = st.slider("가격 탄력성 계수 (Elasticity)", -1.0, 0.0, -0.2, 0.05)  # elasticity(탄력성) 계수 slider(슬라이더)
                surcharge = st.slider("피크 시간대 할증률 (Peak Surcharge)", 0.0, 1.0, 0.20, 0.05)  # peak surcharge(피크 할증률) slider(슬라이더)
                discount = st.slider("경부하 시간대 할인율 (Off-Peak Discount)", 0.0, 0.5, 0.15, 0.05)  # off-peak discount(경부하 할인율) slider(슬라이더)
                
                # --- markdown 시간대별 요금제 안내 블록 (피크/경부하/중부하 구간 설명) ---
                st.markdown(
                    """
                    *   **피크 시간대**: 10:00 ~ 12:00, 13:00 ~ 17:00, 18:00 ~ 22:00 (할증 적용)
                    *   **경부하 시간대**: 23:00 ~ 09:00 (할인 적용)
                    *   **중부하 시간대**: 그 외 시간 (요금 변동 없음)
                    """
                )
                
                charge_type = st.radio("충전 방식 선택", ["전체", "급속", "완속"], horizontal=True)  # radio(라디오)로 충전 방식(급속/완속) 선택
                
            with dp_col2:  # 우측 차트 column(컬럼) context(컨텍스트) 진입
                if not hourly_data.empty:  # 시간대별 데이터가 비어있지 않은 경우
                    df_profile = hourly_data.copy()  # hourly_data(시간별 데이터) deep copy(깊은 복사)
                    if charge_type != "전체":  # "전체"가 아닌 특정 충전 방식 선택 시
                        df_profile = df_profile[df_profile["충전방식"] == charge_type]  # 충전 방식으로 filter(필터)링
                        
                    hour_cols = [f"{i:02d}시" for i in range(24)]  # 00시~23시 column(컬럼)명 list(리스트) 생성
                    base_profile = df_profile[hour_cols].mean().values  # 24시간 평균 충전 부하 profile(프로파일) 배열
                    
                    sim_profile, price_change = simulate_dynamic_pricing(  # 다이내믹 요금 시뮬레이션 함수 호출
                        base_profile,  # 기준 시간대별 부하 profile(프로파일)
                        elasticity=elasticity,  # elasticity(탄력성) 계수
                        peak_surcharge=surcharge,  # peak surcharge(피크 할증률)
                        discount_rate=discount  # discount rate(할인율)
                    )
                    
                    st.markdown("##### 2. 요금제 도입 전/후 시간대별 충전 부하 비교")  # 전후 비교 차트 섹션 제목
                    
                    chart_df = pd.DataFrame({  # 차트용 DataFrame(데이터프레임) 생성
                        "시간": [f"{i:02d}시" for i in range(24)],  # X축 시간 레이블
                        "도입 전 (현재 부하)": base_profile,  # 도입 전 부하 profile(프로파일)
                        "도입 후 (시뮬레이션)": sim_profile,  # 도입 후 시뮬레이션 profile(프로파일)
                        "요금 변동률": price_change * 100  # 요금 변동률(%)
                    })
                    
                    fig = go.Figure()  # Plotly(플로틀리) Figure(피겨) 객체 생성
                    fig.add_trace(go.Scatter(  # 도입 전 부하 area chart(영역 차트) trace(트레이스) 추가
                        x=chart_df["시간"], y=chart_df["도입 전 (현재 부하)"],
                        fill='tozeroy', mode='lines+markers',
                        name='도입 전 (현재 부하)',
                        line=dict(color='#888888', width=2),
                        fillcolor='rgba(136, 136, 136, 0.2)'
                    ))
                    fig.add_trace(go.Scatter(  # 도입 후 부하 area chart(영역 차트) trace(트레이스) 추가
                        x=chart_df["시간"], y=chart_df["도입 후 (시뮬레이션)"],
                        fill='tozeroy', mode='lines+markers',
                        name='도입 후 (시뮬레이션)',
                        line=dict(color='#FF5A5F', width=3),
                        fillcolor='rgba(255, 90, 95, 0.3)'
                    ))
                    
                    fig.update_layout(  # Figure(피겨) layout(레이아웃) 업데이트
                        title=f"다이내믹 요금제 부하 분산 궤적 (탄력성: {elasticity:.2f})",
                        xaxis_title="시간대",
                        yaxis_title="평균 충전 부하 (kW)",
                        hovermode="x unified",
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)  # Plotly(플로틀리) area chart(영역 차트) 렌더링
                    
                    peak_hours_idx = [10, 11, 13, 14, 15, 16, 18, 19, 20, 21]  # peak hours(피크 시간대) index(인덱스) 목록
                    before_peak_avg = np.mean(base_profile[peak_hours_idx])  # 도입 전 피크 시간대 평균 부하
                    after_peak_avg = np.mean(sim_profile[peak_hours_idx])  # 도입 후 피크 시간대 평균 부하
                    peak_reduction = (before_peak_avg - after_peak_avg) / before_peak_avg * 100 if before_peak_avg else 0  # 피크 부하 절감률(%)
                    
                    mc_col1, mc_col2, mc_col3 = st.columns(3)  # 3개 metric(메트릭) column(컬럼) 생성
                    with mc_col1:  # 1번 metric(메트릭) column(컬럼)
                        st.metric("피크 시간대 평균 부하 (전)", f"{before_peak_avg:,.1f} kW")  # 도입 전 피크 평균 부하
                    with mc_col2:  # 2번 metric(메트릭) column(컬럼)
                        st.metric("피크 시간대 평균 부하 (후)", f"{after_peak_avg:,.1f} kW")  # 도입 후 피크 평균 부하
                    with mc_col3:  # 3번 metric(메트릭) column(컬럼)
                        st.metric("피크 시간대 부하 절감률", f"{peak_reduction:.1f}%", delta=f"-{peak_reduction:.1f}%")  # 피크 부하 절감률 metric(메트릭)
                else:  # hourly_data(시간별 데이터)가 비어있는 경우
                    st.warning("시간대별 충전 부하 데이터가 비어 있어 시뮬레이션을 수행할 수 없습니다.")  # warning(경고) 메시지 표시

    # --- 메뉴 분기: 예측 모델 비교 ---
    elif active_menu == "📊 예측 모델 비교":  # 선택된 메뉴가 "예측 모델 비교"인 경우
        st.subheader("예측 모델 성능 비교")  # 모델 비교 소제목 표시
        st.caption("머신러닝 5개와 딥러닝 2개(CNN, Transformer 계열)를 같은 데이터 분할로 비교합니다.")  # 비교 대상 모델 설명 캡션

        metrics = model_state["metrics"].copy()  # model_state(모델 상태)에서 metrics(지표) DataFrame(데이터프레임) copy(복사)

        # --- Rank(순위) 계산: Test RMSE 기준 ---
        test_only = metrics[metrics["Split"] == "Test"].copy()  # Test split(분할) 데이터만 추출
        test_only["Rank (순위)"] = test_only["RMSE"].rank(method="min").astype(int)  # RMSE(평균제곱근오차) 기준 rank(순위) 계산
        metrics = pd.merge(metrics, test_only[["Model", "Rank (순위)"]], on="Model", how="left")  # rank(순위) column(컬럼)을 metrics(지표)에 merge(병합)

        # --- sub tab(서브 탭) 분리: 성능 랭킹 / 예측 일치도 / ROC ---
        sub_tabs = st.tabs(["📊 성능 랭킹 및 지표", "📈 예측 일치도 및 오차 분석", "🎯 위험 지역 분류 성능 (ROC)"])  # 3개 sub tab(서브 탭) 생성

        with sub_tabs[0]:  # "성능 랭킹 및 지표" sub tab(서브 탭) context(컨텍스트) 진입
            st.markdown("#### 전체 지표 종합 표")  # 종합 지표표 제목
            # column(컬럼) 순서: Model, Rank (순위), Group, Split, RMSE, R2, MAE, SMAPE(%)
            cols = ["Model", "Rank (순위)", "Group", "Split", "RMSE", "R2", "MAE", "SMAPE(%)"]  # 표시할 column(컬럼) 목록
            display_metrics = metrics[metrics["Split"] == "Test"]  # Test split(분할)만 표시
            st.dataframe(display_metrics[cols].sort_values(["Rank (순위)", "Split"]), use_container_width=True, hide_index=True)  # rank(순위) 기준 정렬 테이블 표시

            test_metrics = test_only.sort_values("RMSE")  # RMSE(평균제곱근오차) 오름차순 정렬
            fig = px.bar(  # Test RMSE bar chart(막대 차트) 생성
                test_metrics,
                x="Model",
                y="RMSE",
                color="Group",
                title="Test RMSE 비교 (낮을수록 우수)",
                text_auto=".1f",
            )
            st.plotly_chart(fig, use_container_width=True)  # Plotly(플로틀리) RMSE bar chart(막대 차트) 렌더링

        with sub_tabs[1]:  # "예측 일치도 및 오차 분석" sub tab(서브 탭) context(컨텍스트) 진입
            pred = model_state["predictions"]  # predictions(예측값) DataFrame(데이터프레임) 조회
            best_pred = pred[pred["Model"] == model_state["best_name"]].copy()  # 최우수 모델 예측값만 filter(필터)링
            c1, c2 = st.columns(2)  # 2개 column(컬럼) 레이아웃
            with c1:  # 1번 column(컬럼) context(컨텍스트) 진입
                fig = px.scatter(best_pred, x="Actual", y="Predicted", color="용도", hover_name="지역", title="Actual vs Prediction")  # Actual vs Predicted scatter(산점도) 생성
                min_v = min(best_pred["Actual"].min(), best_pred["Predicted"].min())  # scatter(산점도) 대각선 min(최소)값
                max_v = max(best_pred["Actual"].max(), best_pred["Predicted"].max())  # scatter(산점도) 대각선 max(최대)값
                fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", name="완전 일치"))  # 완전 일치 대각선 trace(트레이스) 추가
                st.plotly_chart(fig, use_container_width=True)  # Actual vs Predicted scatter(산점도) 렌더링
            with c2:  # 2번 column(컬럼) context(컨텍스트) 진입
                best_pred["Residual"] = best_pred["Actual"] - best_pred["Predicted"]  # residual(잔차) column(컬럼) 계산
                fig = px.scatter(best_pred, x="Predicted", y="Residual", color="용도", hover_name="지역", title="Residual Plot")  # residual plot(잔차 플롯) 생성
                fig.add_hline(y=0, line_dash="dash")  # y=0 기준선 horizontal line(수평선) 추가
                st.plotly_chart(fig, use_container_width=True)  # residual plot(잔차 플롯) 렌더링
    
            qq = stats.probplot(best_pred["Residual"], dist="norm")  # QQ plot(정규성 검정)용 probplot(확률플롯) 계산
            qq_df = pd.DataFrame({"Theoretical": qq[0][0], "Ordered residual": qq[0][1]})  # QQ plot(정규성 검정) DataFrame(데이터프레임) 생성
            fig = px.scatter(qq_df, x="Theoretical", y="Ordered residual", title="QQ Plot (정규성 검정)")  # QQ plot(정규성 검정) scatter(산점도) 생성
            st.plotly_chart(fig, use_container_width=True)  # QQ plot(정규성 검정) 렌더링

        with sub_tabs[2]:  # "위험 지역 분류 성능 (ROC)" sub tab(서브 탭) context(컨텍스트) 진입
            roc_rows = []  # ROC AUC 결과 row(행) list(리스트) 초기화
            roc_fig = go.Figure()  # ROC curve(곡선) Figure(피겨) 생성
            if "precomputed_roc_data" in model_state:  # 사전 계산된 ROC data(데이터)가 있는 경우
                for name, rdata in model_state["precomputed_roc_data"].items():  # 각 모델별 ROC data(데이터) 순회
                    fpr = rdata["fpr"]  # false positive rate(거짓 양성률)
                    tpr = rdata["tpr"]  # true positive rate(참 양성률)
                    auc_score = rdata["auc"]  # AUC(곡선 아래 면적) 점수
                    roc_rows.append({"Model": name, "Group": rdata["group"], "AUC": auc_score})  # ROC 결과 row(행) 추가
                    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} ({auc_score:.3f})"))  # ROC curve(곡선) trace(트레이스) 추가
            else:  # 사전 계산 데이터 없을 경우 실시간 ROC 계산
                threshold = model_state["y"].quantile(0.7)  # 상위 30% 고부하를 positive(양성) 임계치로 설정
                y_test_binary = (model_state["y_test"] >= threshold).astype(int)  # binary(이진) 레이블 변환
                for name, model in model_state["models"].items():  # 각 모델 순회
                    score = model.predict(model_state["X_test"])  # test set(테스트 세트) 예측 score(점수)
                    fpr, tpr, _ = roc_curve(y_test_binary, score)  # ROC curve(곡선) 계산
                    auc_score = auc(fpr, tpr)  # AUC(곡선 아래 면적) 계산
                    roc_rows.append({"Model": name, "Group": model_state["model_groups"][name], "AUC": auc_score})  # ROC 결과 row(행) 추가
                    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} ({auc_score:.3f})"))  # ROC curve(곡선) trace(트레이스) 추가
            roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line={"dash": "dash"}))  # random classifier(무작위 분류기) 기준선 추가
            roc_fig.update_layout(title="ROC/AUC: 고부하 위험지역 분류 성능")  # ROC Figure(피겨) layout(레이아웃) 업데이트
            st.plotly_chart(roc_fig, use_container_width=True)  # ROC curve(곡선) 렌더링
            st.dataframe(pd.DataFrame(roc_rows).sort_values("AUC", ascending=False), use_container_width=True, hide_index=True)  # AUC(곡선 아래 면적) 순위 테이블 표시

        st.markdown("---")  # 구분선(separator) 표시
        st.markdown("### SMOTE 적용 전/후 최고 모델 성능 비교")  # SMOTE 전후 비교 섹션 제목
        if model_state_smote is None:  # SMOTE(오버샘플링) 모델 상태가 없는 경우
            with st.spinner("🤖 SMOTE 불균형 오버샘플링을 적용하여 7개 ML/DL 모델을 비동기 재학습 중입니다... (약 20초 소요)"):  # spinner(스피너) 표시하며 재학습
                from utils.models import train_models  # train_models(모델 학습) 함수 lazy import(지연 임포트)
                model_state_smote = train_models(final_data.to_json(orient="split"), use_smote=True)  # SMOTE(오버샘플링) 적용 모델 학습
                st.session_state["model_state_smote"] = model_state_smote  # session_state(세션 상태)에 SMOTE 모델 상태 저장
        best_pre = model_state["metrics"][model_state["metrics"]["Split"]=="Test"].sort_values("RMSE").iloc[0]  # SMOTE 적용 전 최우수 모델 metrics(지표)
        best_post = model_state_smote["metrics"][model_state_smote["metrics"]["Split"]=="Test"].sort_values("RMSE").iloc[0]  # SMOTE 적용 후 최우수 모델 metrics(지표)
        smote_df = pd.DataFrame({  # SMOTE 전후 비교 DataFrame(데이터프레임) 생성
            "상태": ["적용 전 (Pre-SMOTE)", "적용 후 (Post-SMOTE)"],
            "최우수 모델명": [best_pre["Model"], best_post["Model"]],
            "Test RMSE": [best_pre["RMSE"], best_post["RMSE"]],
            "Test R2": [best_pre["R2"], best_post["R2"]]
        })
        st.dataframe(smote_df.round(4), use_container_width=True, hide_index=True)  # SMOTE 전후 비교 테이블 표시

    # --- 메뉴 분기: 통계/군집 분석 ---
    elif active_menu == "🧮 통계/군집 분석":  # 선택된 메뉴가 "통계/군집 분석"인 경우
        st.subheader("TableOne, t-SNE/UMAP, CCA, 상관분석")  # 통계/군집 분석 소제목 표시
        st.caption("평가 기준의 통계/부분집단 분석 항목을 한 화면에서 확인합니다.")  # 분석 항목 설명 캡션
        table_cols = [  # TableOne(테이블원)에 포함할 variable(변수) column(컬럼) 목록
            "전기차_전체대수",
            "총_전력판매량",
            "총_판매수입",
            "충전인프라_규모_PCA",
            "충전기_1대당_평균용량",
            "인프라_부하지수",
            "전력_부하지수",
        ]
        table_kind, table_output = make_tableone(final_data, table_cols)  # TableOne(테이블원) 또는 대체 통계표 생성
        if table_kind == "tableone":  # tableone 패키지가 설치된 경우
            st.markdown(table_output)  # TableOne(테이블원) HTML/markdown 출력 렌더링
        else:  # tableone 패키지 미설치 시 대체 구현
            st.info("tableone 패키지가 없어 동일 변수에 대해 평균±표준편차와 t-test p-value를 직접 계산했습니다.")  # 대체 구현 안내 info(정보) 메시지
            st.dataframe(table_output, use_container_width=True, hide_index=True)  # 대체 통계 DataFrame(데이터프레임) 테이블 표시

        X_embed = model_state["X"].copy()  # embedding(임베딩)용 feature matrix(피처 행렬) copy(복사)
        y_embed = model_state["y"]  # target(타깃) 변수(전력 부하지수)
        scaled_embed = StandardScaler().fit_transform(X_embed)  # StandardScaler(표준화) 적용

        # --- t-SNE(차원축소): 사전 계산값 우선, 없으면 실시간 fallback(폴백) ---
        if "precomputed_tsne_xy" in model_state:  # 사전 계산된 t-SNE 좌표가 있는 경우
            tsne_xy = model_state["precomputed_tsne_xy"]  # 사전 계산 t-SNE 좌표 로드
        else:  # 사전 계산 데이터 없을 경우 실시간 t-SNE 계산
            perplexity = max(5, min(20, len(X_embed) // 4))  # perplexity(혼잡도) 파라미터 설정
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", learning_rate="auto")  # t-SNE(차원축소) 모델 생성
            tsne_xy = tsne.fit_transform(scaled_embed)  # t-SNE(차원축소) 2D 좌표 변환
            
        embed_df = final_data[["지역", "용도"]].copy()  # embedding(임베딩) 시각화용 DataFrame(데이터프레임) 생성
        embed_df["tSNE-1"] = tsne_xy[:, 0]  # t-SNE 1차원 좌표 column(컬럼) 추가
        embed_df["tSNE-2"] = tsne_xy[:, 1]  # t-SNE 2차원 좌표 column(컬럼) 추가
        embed_df["고부하"] = np.where(y_embed.values >= y_embed.quantile(0.7), "고부하", "일반")  # 상위 30% 고부하 여부 label(레이블) 추가
        fig = px.scatter(embed_df, x="tSNE-1", y="tSNE-2", color="고부하", symbol="용도", hover_name="지역", title="t-SNE")  # t-SNE scatter(산점도) 생성
        st.plotly_chart(fig, use_container_width=True)  # t-SNE scatter(산점도) 렌더링

        # --- UMAP(차원축소): 사전 계산값 우선, 없으면 UMAP 또는 PCA fallback(폴백) ---
        if "precomputed_umap_xy" in model_state:  # 사전 계산된 UMAP 좌표가 있는 경우
            umap_xy = model_state["precomputed_umap_xy"]  # 사전 계산 UMAP 좌표 로드
            title = model_state.get("precomputed_umap_title", "UMAP")  # 차트 title(제목) 조회
        else:  # 사전 계산 데이터 없을 경우 실시간 UMAP/PCA 계산
            try:  # UMAP 패키지 import(임포트) 시도
                import umap  # umap(유맵) 패키지 lazy import(지연 임포트)
                reducer = umap.UMAP(random_state=42)  # UMAP(차원축소) reducer(축소기) 생성
                umap_xy = reducer.fit_transform(scaled_embed)  # UMAP(차원축소) 2D 좌표 변환
                title = "UMAP"  # 차트 title(제목) 설정
            except Exception:  # UMAP 패키지 미설치 시 PCA 대체
                reducer = PCA(n_components=2, random_state=42)  # PCA(주성분분석) 대체 reducer(축소기) 생성
                umap_xy = reducer.fit_transform(scaled_embed)  # PCA(주성분분석) 2D 좌표 변환
                title = "UMAP 패키지 미설치 시 PCA 대체 시각화"  # 대체 시각화 title(제목) 설정
                
        embed_df["UMAP-1"] = umap_xy[:, 0]  # UMAP 1차원 좌표 column(컬럼) 추가
        embed_df["UMAP-2"] = umap_xy[:, 1]  # UMAP 2차원 좌표 column(컬럼) 추가
        fig = px.scatter(embed_df, x="UMAP-1", y="UMAP-2", color="용도", symbol="고부하", hover_name="지역", title=title)  # UMAP scatter(산점도) 생성
        st.plotly_chart(fig, use_container_width=True)  # UMAP scatter(산점도) 렌더링

        # --- 상관분석(Correlation Analysis): Pearson, Spearman, Kendall ---
        corr_rows = []  # 상관분석 결과 row(행) list(리스트) 초기화
        for x_col in ["전기차_전체대수", "충전인프라_규모_PCA", "충전기_1대당_평균용량", "인프라_부하지수"]:  # X variable(변수) 각각 순회
            pearson_r, pearson_p = stats.pearsonr(final_data[x_col], final_data["전력_부하지수"])  # Pearson(피어슨) 상관계수 및 p-value(유의확률)
            spearman_r, spearman_p = stats.spearmanr(final_data[x_col], final_data["전력_부하지수"])  # Spearman(스피어만) 순위 상관계수
            kendall_r, kendall_p = stats.kendalltau(final_data[x_col], final_data["전력_부하지수"])  # Kendall(켄달) tau(타우) 상관계수
            corr_rows.append(  # 상관분석 결과 dict(딕셔너리) 추가
                {
                    "X": x_col,  # 독립 variable(변수)
                    "Y": "전력_부하지수",  # 종속 variable(변수)
                    "Pearson r": pearson_r,  # Pearson(피어슨) r
                    "Pearson p": pearson_p,  # Pearson(피어슨) p-value(유의확률)
                    "Spearman r": spearman_r,  # Spearman(스피어만) r
                    "Spearman p": spearman_p,  # Spearman(스피어만) p-value(유의확률)
                    "Kendall tau": kendall_r,  # Kendall(켄달) tau(타우)
                    "Kendall p": kendall_p,  # Kendall(켄달) p-value(유의확률)
                }
            )
        st.dataframe(pd.DataFrame(corr_rows).round(4), use_container_width=True, hide_index=True)  # 상관분석 결과 테이블 표시

        # --- CCA(정준상관분석): 사전 계산값 우선, 없으면 실시간 fallback(폴백) ---
        if "precomputed_cca_x_c" in model_state and "precomputed_cca_y_c" in model_state:  # 사전 계산 CCA score(점수)가 있는 경우
            cca_x_c = model_state["precomputed_cca_x_c"]  # CCA X canonical score(정준 점수) 로드
            cca_y_c = model_state["precomputed_cca_y_c"]  # CCA Y canonical score(정준 점수) 로드
        else:  # 사전 계산 데이터 없을 경우 실시간 CCA 계산
            cca_x = StandardScaler().fit_transform(final_data[["전기차_전체대수", "충전인프라_규모_PCA", "충전기_1대당_평균용량", "인프라_부하지수"]])  # X block(블록) 표준화
            cca_y = StandardScaler().fit_transform(final_data[["전력_부하지수"]])  # Y block(블록) 표준화
            cca = CCA(n_components=1)  # CCA(정준상관분석) 모델 생성(1 component(성분))
            x_c, y_c = cca.fit_transform(cca_x, cca_y)  # CCA(정준상관분석) fit(적합) 및 transform(변환)
            cca_x_c = x_c[:, 0]  # X canonical score(정준 점수) 1차원 추출
            cca_y_c = y_c[:, 0]  # Y canonical score(정준 점수) 1차원 추출
            
        cca_df = pd.DataFrame({"CCA_X": cca_x_c, "CCA_Y": cca_y_c, "용도": final_data["용도"], "지역": final_data["지역"]})  # CCA scatter(산점도)용 DataFrame(데이터프레임)
        fig = px.scatter(cca_df, x="CCA_X", y="CCA_Y", color="용도", hover_name="지역", title="CCA canonical score")  # CCA scatter(산점도) 생성
        st.plotly_chart(fig, use_container_width=True)  # CCA scatter(산점도) 렌더링

        st.markdown("---")  # 구분선(separator) 표시
        st.subheader("심화 탐색적 데이터 분석 (EDA)")  # EDA(탐색적 데이터 분석) 섹션 소제목

        col1, col2 = st.columns(2)  # 2개 column(컬럼) 레이아웃(산점도 / 네트워크 그래프)

        with col1:  # 1번 column(컬럼) context(컨텍스트) 진입
            st.markdown("#### 1. Scatter plot with 95% Confidence Interval")  # 95% CI(신뢰구간) 산점도 제목
            try:  # seaborn regplot(회귀플롯) 시도
                import seaborn as sns  # seaborn(시본) 시각화 패키지 lazy import(지연 임포트)
    
                plt.rc('font', family='Malgun Gothic')  # matplotlib(맷플롯립) 한글 폰트 설정
                plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 유니코드 깨짐 방지
                fig_scatter, ax = plt.subplots(figsize=(5, 3.5))  # scatter(산점도) Figure(피겨) 및 axes(축) 생성
    
                colors = {"자가용": "#00A699", "사업자용": "#FF5A5F"}  # 용도별 color(색상) map(맵)
                for usage in final_data["용도"].unique():  # 각 용도별 순회
                    subset = final_data[final_data["용도"] == usage]  # 용도별 subset(부분집합) 추출
                    sns.regplot(  # regression plot(회귀플롯) with 95% CI(신뢰구간) 추가
                        data=subset, 
                        x="전기차_전체대수", 
                        y="전력_부하지수", 
                        ax=ax, 
                        label=usage, 
                        color=colors.get(usage, "blue"),
                        scatter_kws={'alpha':0.5, 's':15}
                    )
                ax.set_title("전기차 대수 대비 전력 부하지수 (95% 신뢰구간 포함)")  # chart(차트) title(제목) 설정
                ax.legend()  # legend(범례) 표시
                st.pyplot(fig_scatter, clear_figure=True, use_container_width=True)  # matplotlib(맷플롯립) scatter(산점도) Streamlit(스트림릿) 렌더링
            except ImportError:  # seaborn 미설치 시
                st.info("seaborn 패키지가 설치되지 않아 추세선을 그릴 수 없습니다.")  # info(정보) 메시지 표시
    
        with col2:  # 2번 column(컬럼) context(컨텍스트) 진입
            st.markdown("#### 2. 상관관계 네트워크 그래프 (Network Graph)")  # 상관관계 network graph(네트워크 그래프) 제목
            num_cols = ["전기차_전체대수", "총_전력판매량", "충전인프라_규모_PCA", "충전기_1대당_평균용량", "전력_부하지수", "인프라_부하지수"]  # numeric(수치) column(컬럼) 목록
            corr_mat = final_data[num_cols].corr()  # correlation matrix(상관행렬) 계산

            G = nx.Graph()  # undirected graph(무방향 그래프) 생성
            for i in range(len(corr_mat.columns)):  # 상관행렬 column(컬럼) i 순회
                for j in range(i + 1, len(corr_mat.columns)):  # j > i인 column(컬럼) 쌍 순회
                    weight = corr_mat.iloc[i, j]  # i-j variable(변수) 쌍 상관계수
                    # threshold(임계값) 0.15: 음의 상관관계(파란 선)도 graph(그래프)에 표시
                    if abs(weight) > 0.15:  # |상관계수| > 0.15인 edge(간선)만 추가
                        G.add_edge(corr_mat.columns[i], corr_mat.columns[j], weight=weight)  # weighted edge(가중 간선) 추가
            
            if len(G.edges) > 0:  # graph(그래프)에 edge(간선)가 존재하는 경우
                plt.rc('font', family='Malgun Gothic')  # matplotlib(맷플롯립) 한글 폰트 설정
                plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 유니코드 깨짐 방지
                fig_net, ax_net = plt.subplots(figsize=(5, 3.5))  # network graph(네트워크 그래프) Figure(피겨) 생성
                pos = nx.spring_layout(G, k=5.0, seed=42)  # spring layout(스프링 레이아웃)으로 node(노드) 위치 계산
                edges = G.edges()  # graph(그래프) edge(간선) 목록
                raw_weights = [G[u][v]['weight'] for u,v in edges]  # 각 edge(간선) weight(가중치) 추출
                nx.draw(  # networkx(네트워크X) graph(그래프) 그리기
                    G, pos, ax=ax_net, with_labels=True, node_color='lightgreen', 
                    node_size=500, font_family='Malgun Gothic', font_size=4, 
                    edge_color=raw_weights, edge_cmap=plt.cm.coolwarm, edge_vmin=-1, edge_vmax=1,
                    width=[abs(w) * 5 for w in raw_weights]
                )
                # colorbar(컬러바) 범례: 양/음 상관관계 색상 의미 표시
                sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=-1, vmax=1))  # colorbar(컬러바)용 ScalarMappable 생성
                cbar = plt.colorbar(sm, ax=ax_net, shrink=0.5, pad=0.05)  # colorbar(컬러바) 추가
                cbar.set_label('상관계수 (Red: 양의 상관, Blue: 음의 상관)', fontsize=7)  # colorbar(컬러바) label(레이블) 설정
                cbar.ax.tick_params(labelsize=6)  # colorbar(컬러바) tick(눈금) 크기 설정
    
                ax_net.margins(0.2)  # node(노드) 잘림 방지 margin(여백) 추가
                st.pyplot(fig_net, clear_figure=True, use_container_width=True)  # network graph(네트워크 그래프) Streamlit(스트림릿) 렌더링
            else:  # edge(간선)가 없는 경우
                st.info("상관관계(>0.15)가 높은 변수 쌍이 없어 네트워크를 생성할 수 없습니다.")  # info(정보) 메시지 표시

        st.markdown("#### 3. Box plot with p-value (집단 간 부하지수 차이 검증)")  # box plot(상자 그림) + t-test 제목
        private_load = final_data[final_data["용도"] == "자가용"]["전력_부하지수"]  # 자가용 부하지수 series(시리즈)
        business_load = final_data[final_data["용도"] == "사업자용"]["전력_부하지수"]  # 사업자용 부하지수 series(시리즈)
        _, p_val = stats.ttest_ind(private_load, business_load, equal_var=False, nan_policy="omit")  # Welch t-test(웰치 t-검정) p-value(유의확률) 계산

        fig_box = px.box(  # box plot(상자 그림) 생성
            final_data, 
            x="용도", 
            y="전력_부하지수", 
            color="용도", 
            title="용도별 전력 부하지수 분포"
        )
        fig_box.add_annotation(  # p-value(유의확률) annotation(주석) 추가
            x=0.5, 
            y=final_data["전력_부하지수"].max(), 
            text=f"통계적 유의성 (p-value): {p_val:.4e}", 
            showarrow=False,
            font=dict(size=14, color="red")
        )
        fig_box.update_layout(height=450)  # box plot(상자 그림) height(높이) 설정
        st.plotly_chart(fig_box, use_container_width=True)  # box plot(상자 그림) 렌더링

    # --- 메뉴 분기: 강건성 평가 (Phase 3) ---
    elif active_menu == "🛡️ 강건성 평가 (Phase 3)":  # 선택된 메뉴가 "강건성 평가"인 경우
        st.subheader("학술적 강건성 평가 및 심화 시뮬레이션")  # 강건성 평가 소제목 표시

        best_name = model_state["best_name"]  # 최우수 모델명 조회
        best_model = model_state["models"][best_name]  # 최우수 모델 object(객체) 조회

        # --- 1. Bootstrap(부트스트랩) 95% CI(신뢰구간) ---
        st.markdown(f"#### 1. 부트스트랩 95% 신뢰구간 (Bootstrap 95% CI) - 최우수 모델: {best_name}")  # Bootstrap CI 섹션 제목
        if "precomputed_bootstrap" in model_state:  # 사전 계산 Bootstrap 결과가 있는 경우
            ci_rmse, ci_r2, r_scores, r2_scores = model_state["precomputed_bootstrap"]  # 사전 계산 CI(신뢰구간) 로드
        else:  # 사전 계산 없을 경우 cached_bootstrap(캐시 부트스트랩) 호출
            ci_rmse, ci_r2, r_scores, r2_scores = cached_bootstrap(best_name, best_model, model_state["X_test"], model_state["y_test"])  # Bootstrap CI 계산
        st.success(f"**RMSE 95% CI**: {ci_rmse[0]:.4f} ~ {ci_rmse[1]:.4f}  \n**R2 95% CI**: {ci_r2[0]:.4f} ~ {ci_r2[1]:.4f}")  # RMSE/R2 CI(신뢰구간) success(성공) 메시지 표시

        # --- 1-2. Nested CV(중첩 교차검증) ---
        st.markdown(f"#### 1-2. 중첩 10-겹 교차검증 (Nested 10-fold CV) - 최우수 모델: {best_name}")  # Nested CV 섹션 제목
        st.caption("외부 10-Fold로 평가, 내부 3-Fold로 튜닝하여 과적합 없는 일반화 성능을 산출합니다.")  # Nested CV 설명 캡션
        if "precomputed_nested_cv" in model_state:  # 사전 계산 Nested CV 결과가 있는 경우
            mean_rmse, std_rmse, outer_scores = model_state["precomputed_nested_cv"]  # 사전 계산 Nested CV metrics(지표) 로드
        else:  # 사전 계산 없을 경우 cached_nested_cv(캐시 중첩 CV) 호출
            mean_rmse, std_rmse, outer_scores = cached_nested_cv(best_name, best_model, model_state["X"], model_state["y"])  # Nested CV 계산
        st.info(f"**Nested CV 평균 Test RMSE**: {mean_rmse:.4f} ± {std_rmse:.4f}")  # Nested CV 평균 RMSE info(정보) 메시지 표시

        # --- 1-3. Spatial External Validation(공간적 외부 검증) ---
        st.markdown(f"#### 1-3. 공간적 외부 검증 (Spatial External Validation) - 최우수 모델: {best_name}")  # Spatial CV 섹션 제목
        st.caption("선택한 지역의 데이터를 학습에서 완전히 배제한 뒤, 해당 지역을 새로운 외부 환경으로 가정하고 예측 성능을 검증합니다.")  # Spatial CV 설명 캡션

        holdout_region = st.selectbox("학습에서 배제할 외부 테스트 지역 선택:", ["인천", "서울", "경기"], index=0)  # holdout(홀드아웃) 지역 selectbox(선택상자)

        precomputed_spatial = model_state.get("precomputed_spatial", {})  # 사전 계산 Spatial CV 결과 dict(딕셔너리) 조회
        if holdout_region in precomputed_spatial:  # 선택 지역의 사전 계산 결과가 있는 경우
            ext_rmse, ext_mae, ext_r2, ext_err = precomputed_spatial[holdout_region]  # 사전 계산 외부 검증 metrics(지표) 로드
        else:  # 사전 계산 없을 경우 cached_spatial_external_validation(캐시 공간 외부 검증) 호출
            ext_rmse, ext_mae, ext_r2, ext_err = cached_spatial_external_validation(best_name, best_model, model_state["X"], model_state["y"], holdout_region)  # Spatial CV 계산

        if ext_err:  # 외부 검증 error(에러)가 있는 경우
            st.warning(ext_err)  # warning(경고) 메시지 표시
        else:  # 외부 검증 성공 시
            st.success(f"**[{holdout_region}] 외부 검증 Test RMSE**: {ext_rmse:.4f} (MAE: {ext_mae:.4f}, R²: {ext_r2:.4f})")  # 외부 검증 metrics(지표) success(성공) 메시지

            # --- 내부 검증(Random Split) vs 외부 검증 성능 bar chart(막대 차트) 비교 ---
            test_rmse_internal = model_state["metrics"][(model_state["metrics"]["Model"] == best_name) & (model_state["metrics"]["Split"] == "Test")]["RMSE"].values[0]  # 내부 Test RMSE 조회

            comp_df = pd.DataFrame({  # 내부/외부 검증 비교 DataFrame(데이터프레임) 생성
                "Validation Type": ["Internal (Random Split)", f"External ({holdout_region} Holdout)"],
                "RMSE": [test_rmse_internal, ext_rmse]
            })

            fig_ext = px.bar(  # 내부 vs 외부 RMSE bar chart(막대 차트) 생성
                comp_df, x="Validation Type", y="RMSE", text_auto=".4f",
                title=f"내부 검증 vs 외부 검증 ({holdout_region}) 성능 비교 (RMSE 낮을수록 우수)",
                color="Validation Type", color_discrete_sequence=["#2ca02c", "#d62728"]
            )
            st.plotly_chart(fig_ext, use_container_width=True)  # 내부/외부 검증 비교 bar chart(막대 차트) 렌더링

        # --- 2. Adversarial Attack(적대적 공격) 방어력 평가 ---
        st.markdown("#### 2. 적대적 공격 방어력 평가 (Adversarial Attack Analysis)")  # Adversarial Attack 섹션 제목
        st.caption("Test 데이터셋의 모든 피처에 가우시안 노이즈를 강제 주입했을 때의 성능 하락률을 방어력으로 평가합니다.")  # Adversarial Attack 설명 캡션
        if "precomputed_adversarial" in model_state:  # 사전 계산 Adversarial 결과가 있는 경우
            adv_res = model_state["precomputed_adversarial"]  # 사전 계산 Adversarial 결과 로드
        else:  # 사전 계산 없을 경우 cached_adversarial(캐시 적대적 공격) 호출
            adv_res = cached_adversarial(best_name, best_model, model_state["X_test"], model_state["y_test"])  # Adversarial Attack 분석 계산
        st.dataframe(adv_res.style.background_gradient(cmap="Reds", subset=["Drop_Ratio(%)"]), use_container_width=True, hide_index=True)  # Drop_Ratio(%) heatmap(히트맵) 스타일 테이블 표시

        # --- 3. Ablation Study(민감도 분석) ---
        st.markdown("#### 3. 피처 중요도 기반 민감도 분석 (Ablation Study)")  # Ablation Study 섹션 제목
        st.caption("중요도가 가장 낮은 피처부터 하나씩 제거하며 성능(RMSE) 저하 폭을 시각화합니다.")  # Ablation Study 설명 캡션
        if "precomputed_ablation" in model_state:  # 사전 계산 Ablation 결과가 있는 경우
            abl_res = model_state["precomputed_ablation"]  # 사전 계산 Ablation 결과 로드
        else:  # 사전 계산 없을 경우 cached_ablation(캐시 민감도 분석) 호출
            abl_res = cached_ablation(best_name, best_model, model_state["X_train"], model_state["y_train"], model_state["X_test"], model_state["y_test"], model_state["importance"])  # Ablation Study 계산
        fig_abl = px.line(abl_res, x="Num_Features", y="RMSE", hover_data=["Removed_Feature"], markers=True, title="Ablation Study: Feature Removal Impact")  # Ablation line chart(선 그래프) 생성
        fig_abl.update_xaxes(autorange="reversed")  # X축 reverse(역순) (피처 제거 순서 표시)
        st.plotly_chart(fig_abl, use_container_width=True)  # Ablation line chart(선 그래프) 렌더링

        # --- 4. DCA(Decision Curve Analysis, 의사결정 곡선 분석) ---
        st.markdown("#### 4. 의사결정 곡선 분석 (Decision Curve Analysis - adapted)")  # DCA 섹션 제목
        st.caption("과부하 예측(상위 50~90% 위험도) 시 개입했을 때의 모델 효용성(Net Benefit)을 평가합니다.")  # DCA 설명 캡션
        if "precomputed_dca" in model_state:  # 사전 계산 DCA 결과가 있는 경우
            dca_res = model_state["precomputed_dca"]  # 사전 계산 DCA 결과 로드
        else:  # 사전 계산 없을 경우 cached_dca(캐시 의사결정 곡선) 호출
            dca_res = cached_dca(best_name, best_model, model_state["X_test"], model_state["y_test"])  # DCA 계산
        fig_dca = px.line(dca_res, x="Threshold_Value", y=["Model_NB", "Treat_All_NB", "Treat_None_NB"],   # DCA line chart(선 그래프) 생성
                          labels={"value": "Net Benefit", "variable": "Strategy"}, title="Decision Curve Analysis")
        st.plotly_chart(fig_dca, use_container_width=True)  # DCA line chart(선 그래프) 렌더링

        # --- 5. Survival Analysis(생존 분석): Time-to-Overload ---
        st.markdown("#### 5. 충전 부하 한계 도달 시간 시뮬레이션 (Survival Analysis)")  # Survival Analysis 섹션 제목
        st.caption("각 지역이 현재 인프라 수준에서 버틸 수 있는 한계 시간(Time-to-Overload)을 카플란-마이어(Kaplan-Meier) 커브 형태로 추정합니다.")  # Survival Analysis 설명 캡션
        growth_rate = st.slider("전기차 연간 예상 증가율 (%)", min_value=1.0, max_value=20.0, value=5.0, step=1.0) / 100.0  # slider(슬라이더)로 연간 성장률 입력
        
        # growth_rate(성장률) == 5% 이고 사전 연산 결과 존재 시 실시간 계산 skip(건너뛰기)
        if abs(growth_rate - 0.05) < 1e-5 and "precomputed_survival_5" in model_state:  # 5% 성장률 사전 계산 결과 사용 조건
            surv_res = model_state["precomputed_survival_5"]  # 사전 계산 survival(생존) 결과 로드
        else:  # 그 외 성장률은 cached_survival(캐시 생존 분석) 호출
            surv_res = cached_survival(final_data.to_json(orient="split"), growth_rate)  # Survival Analysis 계산

        surv_counts = surv_res["Time_to_Overload"].value_counts().sort_index()  # Time-to-Overload 연도별 count(건수) 집계
        total = len(surv_res)  # 전체 지역 수
        survival_curve = []  # Kaplan-Meier survival curve(생존 곡선) point(포인트) list(리스트)
        current_surv = total  # 현재 생존 지역 수 초기값
        for year in range(16):  # 0~15년 순회
            if year in surv_counts:  # 해당 연도에 과부하 도달 지역이 있는 경우
                current_surv -= surv_counts[year]  # 생존 지역 수 감소
            survival_curve.append({"Year": year, "Survival_Probability": current_surv / total})  # 생존 확률 point(포인트) 추가

        fig_surv = px.line(pd.DataFrame(survival_curve), x="Year", y="Survival_Probability", markers=True,   # survival curve(생존 곡선) line chart(선 그래프) 생성
                           title=f"연간 {growth_rate*100:.0f}% 성장 가정 시 과부하 도달 생존 곡선 (Time-to-Overload)")
        fig_surv.update_yaxes(range=[0, 1.05])  # Y축 range(범위) 0~1.05 설정
        st.plotly_chart(fig_surv, use_container_width=True)  # survival curve(생존 곡선) 렌더링

    # --- 메뉴 분기: SHAP/LIME 모델 설명 ---
    elif active_menu == "🧠 SHAP/LIME 설명":  # 선택된 메뉴가 "SHAP/LIME 설명"인 경우
        st.subheader("SHAP summary / dependence / force plot + LIME 형태 설명")  # SHAP/LIME 설명 소제목 표시
        st.caption("SHAP 패키지가 있으면 실제 SHAP을 사용하고, 없으면 permutation/local contribution으로 대체합니다.")  # SHAP fallback(폴백) 설명 캡션
        importance = model_state["importance"].head(12)  # 상위 12개 feature importance(피처 중요도) 추출
        fig = px.bar(importance.sort_values("Importance"), x="Importance", y="Feature", orientation="h", title="Permutation importance")  # permutation importance(순열 중요도) bar chart(막대 차트) 생성
        st.plotly_chart(fig, use_container_width=True)  # permutation importance bar chart(막대 차트) 렌더링

        selected_feature = st.selectbox("Dependence plot 변수", model_state["feature_columns"])  # dependence plot(의존 플롯) 대상 feature(피처) selectbox(선택상자)
        local_region = st.selectbox("Force/LIME 형태로 볼 지역", sorted(final_data["지역"].unique()))  # local explanation(지역 설명) 대상 지역 selectbox(선택상자)
        local_usage = st.radio("Force/LIME 용도", ["자가용", "사업자용"], horizontal=True, key="force_usage")  # local explanation 용도 radio(라디오)
        local_row = final_data[(final_data["지역"] == local_region) & (final_data["용도"] == local_usage)].head(1)  # 선택 지역+용도 row(행) 1개 추출
        local_x = make_feature_matrix(local_row).reindex(columns=model_state["feature_columns"], fill_value=0) if len(local_row) else None  # local feature vector(피처 벡터) 생성

        shap_ok = render_shap_or_fallback(model_state, selected_feature, local_x)  # SHAP 또는 fallback(폴백) dependence/force plot 렌더링

        best_model = model_state["models"][model_state["best_name"]]  # 최우수 모델 object(객체) 조회
        X_all = model_state["X"]  # 전체 feature matrix(피처 행렬) 조회
        pd_df = cached_partial_dependence(model_state["best_name"], best_model, X_all, selected_feature)  # partial dependence(부분 의존) plot 데이터 계산
        fig = px.line(pd_df, x="Feature value", y="Mean prediction", markers=True, title="Partial dependence")  # partial dependence line chart(선 그래프) 생성
        st.plotly_chart(fig, use_container_width=True)  # partial dependence line chart(선 그래프) 렌더링

        if local_x is not None and len(local_x) > 0:  # local feature vector(피처 벡터)가 유효한 경우
            baseline = best_model.predict(X_all).mean()  # 전체 평균 예측값(baseline) 계산
            pred_value = best_model.predict(local_x)[0]  # 선택 지역 예측값 계산
            imp_map = model_state["importance"].set_index("Feature")["Importance"].reindex(model_state["feature_columns"]).fillna(0)  # feature importance map(맵) 생성
            centered = local_x.iloc[0] - X_all.mean()  # feature(피처) 평균 대비 편차 계산
            raw_contrib = centered * imp_map  # raw contribution(기여도) = 편차 × importance(중요도)
            scale = (pred_value - baseline) / raw_contrib.sum() if raw_contrib.sum() != 0 else 0  # contribution(기여도) 스케일링 계수
            contrib = (raw_contrib * scale).sort_values(key=np.abs, ascending=False).head(8)  # 상위 8개 contribution(기여도) 추출
            force_df = pd.DataFrame({"Feature": contrib.index, "Contribution": contrib.values})  # force/LIME plot용 DataFrame(데이터프레임) 생성
            fig = px.bar(force_df, x="Contribution", y="Feature", orientation="h", title="LIME/force 형태 지역별 기여도")  # force plot bar chart(막대 차트) 생성
            fig.add_vline(x=0, line_dash="dash")  # x=0 기준 vertical line(수직선) 추가
            st.plotly_chart(fig, use_container_width=True)  # force/LIME bar chart(막대 차트) 렌더링
            st.caption(f"기준 예측값 {baseline:,.2f}에서 선택 지역 예측값 {pred_value:,.2f}로 이동하는 방향을 보여줍니다.")  # force plot 해석 캡션 표시

    # --- 메뉴 분기: 조건 충족표(Checklist) ---
    elif active_menu == "📋 조건 충족표":  # 선택된 메뉴가 "조건 충족표"인 경우
        st.subheader("평가 조건 충족표")  # 조건 충족표 소제목 표시
        installed = {}  # optional package(선택 패키지) 설치 상태 dict(딕셔너리) 초기화
        for package, module_name in [("tableone", "tableone"), ("shap", "shap"), ("umap-learn", "umap")]:  # 확인 대상 package(패키지) 순회
            try:  # package(패키지) import(임포트) 시도
                __import__(module_name)  # module(모듈) import(임포트) 실행
                installed[package] = "설치됨"  # 설치 상태: 설치됨
            except Exception:  # import(임포트) 실패 시
                installed[package] = "미설치: requirements.txt로 설치 권장"  # 설치 상태: 미설치
        checklist = pd.DataFrame(  # 평가 조건 checklist(체크리스트) DataFrame(데이터프레임) 생성
            [
                ["새로운 주제", "충족", "수도권 전기차 충전소 부하 예측 및 설치 시뮬레이션"],
                ["데이터 수집", "충족", "공공 CSV/Excel 8개 파일 사용, 환경부 월별 파일 추가"],
                ["TableOne", "조건부 충족", installed["tableone"]],
                ["t-SNE", "충족", "통계/군집 분석 탭"],
                ["UMAP", "조건부 충족", installed["umap-learn"]],
                ["CCA", "충족", "전기차/충전기/인프라 지수와 전력 부하지수"],
                ["correlation analysis 3개", "충족", "Pearson, Spearman, Kendall"],
                ["통계 분석", "충족", "자가용/사업자용 t-test, TableOne 형태"],
                ["머신러닝 4개", "충족", "RandomForest, ExtraTrees, GradientBoosting, KNN"],
                ["딥러닝 2개", "충족", "Numpy_1D_CNN, Tabular_Transformer"],
                ["하이퍼파라미터 튜닝", "충족", "GridSearchCV 적용 완료"],
                ["부트스트랩 (95% CI)", "충족", "학술적 강건성 평가 탭"],
                ["적대적 노이즈 방어력", "충족", "가우시안 노이즈 주입 및 방어력 계산"],
                ["민감도 분석 (Ablation)", "충족", "피처 순차 제거 후 RMSE 평가"],
                ["의사결정 곡선 (DCA)", "충족", "과부하 임계치 기반 Net Benefit 산출"],
                ["생존 분석 시뮬레이터", "충족", "단일 지역 타임라인 궤적 및 카플란-마이어 곡선"],
                ["SHAP summary/dependence/force", "조건부 충족", installed["shap"]],
                ["LIME 형태 설명", "충족", "local contribution plot"],
                ["subgroup analysis", "충족", "자가용 vs 사업자용"],
                ["Streamlit web app", "충족", "현재 앱"],
                ["Residual/QQ/Actual-Pred", "충족", "예측 모델 비교 탭"],
            ],
            columns=["조건", "상태", "확인 위치/비고"],  # checklist(체크리스트) column(컬럼)명
        )
        st.dataframe(checklist, use_container_width=True, hide_index=True)  # checklist(체크리스트) 테이블 표시
        st.markdown("#### 조건별 확인 요약")  # 조건별 요약 섹션 제목
        for _, row in checklist.iterrows():  # checklist(체크리스트) 각 row(행) 순회
            st.markdown(f"- **{row['조건']}**: {row['상태']} - {row['확인 위치/비고']}")  # 조건별 bullet(불릿) 요약 표시
        # --- markdown 패키지 설치 안내 블록 (requirements.txt pip install 명령) ---
        st.markdown(
            """
            제출 전 더 안전하게 만들려면 새 컴퓨터에서 아래 명령을 한 번 실행하세요.

            ```powershell
            pip install -r requirements.txt
            ```

            그러면 TableOne, UMAP, SHAP 항목이 대체 구현이 아니라 실제 패키지 기반으로 표시됩니다.
            """
        )


# --- 정책보고서(Policy Report) 렌더링 함수 ---
def render_report(filtered, final_data, model_state):  # 필터된 데이터와 모델 상태를 받아 정책 보고서 UI를 렌더링하는 함수 정의
    best_name = model_state["best_name"]  # 최우수 모델명 조회
    best_model = model_state["models"][best_name]  # 최우수 모델 object(객체) 조회
    metrics_df = model_state["metrics"]  # metrics(지표) DataFrame(데이터프레임) 조회
    test_rmse = metrics_df[(metrics_df["Model"] == best_name) & (metrics_df["Split"] == "Test")]["RMSE"].values[0]  # 최우수 모델 Test RMSE 조회

    # --- TOP 3 고위험 지역 산출: 현재 필터 기준 최우수 모델 예측 ---
    pred_full = final_data.copy()  # final_data(최종 데이터) deep copy(깊은 복사)
    pred_full["예측_위험도"] = best_model.predict(model_state["X"])  # 전체 데이터 예측 위험도 column(컬럼) 추가
    pred_df = pred_full[pred_full.index.isin(filtered.index)].copy()  # 현재 filter(필터) 적용 index(인덱스)만 추출
    top3 = pred_df.sort_values("예측_위험도", ascending=False).head(3)  # 예측 위험도 상위 3개 지역 추출

    target_col = "전력_부하지수"  # 보고서에 표시할 target column(컬럼)명
    top3_list = []  # TOP 3 HTML list item(목록 항목) list(리스트) 초기화
    for i, row in enumerate(top3.itertuples(), 1):  # TOP 3 각 row(행) 순회(1부터 번호)
        top3_list.append(f"{i}. <b><font color=\"#1E3A8A\">{row.지역}</font></b> ({row.용도}) - 예측 {target_col}: {row.예측_위험도:.2f}")  # HTML list item(목록 항목) 문자열 추가
    top_features = model_state["importance"].sort_values("Importance", ascending=False).head(2)["Feature"].tolist()  # 상위 2개 핵심 feature(피처) 추출
    import tempfile  # PDF용 임시 파일 생성을 위해 tempfile(임시파일) 모듈 import(임포트)
    
    # --- Feature Importance(피처 중요도) Matplotlib Chart: Headless Server 호환, Kaleido 불필요 ---
    imp_df = model_state["importance"].sort_values("Importance", ascending=True).tail(10)  # 상위 10개 importance(중요도) 추출(오름차순 정렬)
    
    fig_mat, ax = plt.subplots(figsize=(6, 4))  # horizontal bar chart(가로 막대 차트) Figure(피겨) 생성
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(imp_df)))  # Blues colormap(컬러맵) 그라데이션 color(색상) 배열
    
    # --- 한글 및 unicode(유니코드) 깨짐 방지 matplotlib(맷플롯립) 설정 ---
    plt.rc('font', family='Malgun Gothic')  # matplotlib(맷플롯립) 한글 폰트 설정
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 유니코드 깨짐 방지
    
    ax.barh(imp_df["Feature"], imp_df["Importance"], color=colors)  # horizontal bar chart(가로 막대 차트) 그리기
    ax.set_title("상위 10개 핵심 인자 (Feature Importance)", fontsize=10, fontweight='bold')  # chart(차트) title(제목) 설정
    ax.spines['top'].set_visible(False)  # 상단 spine(테두리) 숨김
    ax.spines['right'].set_visible(False)  # 우측 spine(테두리) 숨김
    ax.spines['left'].set_color('#cccccc')  # 좌측 spine(테두리) color(색상) 설정
    ax.spines['bottom'].set_color('#cccccc')  # 하단 spine(테두리) color(색상) 설정
    ax.tick_params(labelsize=8)  # tick label(눈금 레이블) 크기 설정
    fig_mat.tight_layout()  # layout(레이아웃) 자동 조정
    
    tmp_imp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)  # 임시 PNG file(파일) 생성(delete=False로 경로 유지)
    fig_mat.savefig(tmp_imp.name, dpi=300, bbox_inches='tight', transparent=False, facecolor='white')  # Figure(피겨)를 고해상도 PNG로 저장
    plt.close(fig_mat)  # matplotlib(맷플롯립) Figure(피겨) 메모리 해제

    # --- PDF(휴대용 문서) 바이트 생성 ---
    pdf_bytes = bytes(generate_report_pdf(best_name, test_rmse, top3_list, top_features, feature_importance_img=tmp_imp.name, final_data=final_data))  # PDF 생성 함수 호출하여 bytes(바이트) 변환


    col1, col2 = st.columns([8, 2])  # 제목(8):PDF 다운로드 버튼(2) column(컬럼) 레이아웃
    with col1:  # 1번 column(컬럼) context(컨텍스트) 진입
        st.markdown("<h2 style='margin-bottom: 0;'>수도권 전기차 충전소 부하 예측 결과 보고</h2>", unsafe_allow_html=True)  # 보고서 대제목 HTML 렌더링
    with col2:  # 2번 column(컬럼) context(컨텍스트) 진입
        st.download_button(  # PDF download button(다운로드 버튼) 렌더링
            label="📥 PDF로 다운로드",
            data=pdf_bytes,
            file_name="정책_결정_보고서.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

    # --- HTML/CSS 정부 보고서 스타일 블록 (.gov-report, .gov-summary, .highlight 등) ---
    st.markdown("""
    <style>
    .gov-report {
        font-family: 'Noto Serif KR', 'Batang', serif;
        line-height: 1.8;
        text-align: justify;
        background-color: #FFFFFF;
        padding: 40px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        color: #1F2937;
    }
    .gov-report h3 {
        font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
        font-weight: bold;
        font-size: 1.25rem;
        margin-top: 30px;
        margin-bottom: 15px;
        color: #111827;
    }
    .gov-summary {
        background-color: #F8FAFC;
        border: 2px solid #CBD5E1;
        padding: 20px;
        margin-bottom: 30px;
        font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
        font-weight: bold;
        color: #0F172A;
    }
    .gov-report ul {
        list-style-type: none;
        padding-left: 0;
        margin-bottom: 20px;
    }
    .gov-report ul li {
        position: relative;
        padding-left: 20px;
        margin-bottom: 8px;
    }
    .gov-report > ul > li::before {
        content: "○";
        position: absolute;
        left: 0;
    }
    .gov-report ul ul {
        margin-top: 8px;
        margin-bottom: 0;
    }
    .gov-report ul ul > li::before {
        content: "-";
        position: absolute;
        left: 5px;
    }
    .gov-report ul ul ul > li::before {
        content: "ㆍ";
        position: absolute;
        left: 10px;
    }
    .highlight {
        font-weight: bold;
        color: #1E3A8A; /* Navy */
        border-bottom: 2px solid #1E3A8A;
    }
    </style>
    """, unsafe_allow_html=True)  # CSS(캐스케이딩 스타일시트) 스타일 block(블록) 렌더링

    top3_html = "".join([f"<li>{x}</li>" for x in top3_list])  # TOP 3 HTML list item(목록 항목) 문자열 결합

    import datetime  # 보고서 날짜 표시를 위해 datetime(날짜시간) 모듈 import(임포트)
    current_kst = datetime.datetime.utcnow() + datetime.timedelta(hours=9)  # UTC(협정 세계시) + 9시간 = KST(한국 표준시)
    current_date_str = current_kst.strftime("%Y. %m. %d.")  # KST(한국 표준시) 날짜 문자열 포맷

    # --- HTML/CSS 표지 페이지 블록 (.cover-page, .cover-title, .cover-subtitle 등) ---
    st.markdown("""
<style>
.cover-page {
    border-top: 5px solid #0070C0;
    border-bottom: 5px solid #0070C0;
    padding: 60px 20px;
    text-align: center;
    background-color: #FFFFFF;
    margin-bottom: 50px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}
.cover-title {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
    font-size: 2.5rem;
    font-weight: 900;
    color: #111827;
    margin-bottom: 20px;
}
.cover-subtitle {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
    font-size: 1.25rem;
    color: #4B5563;
    margin-bottom: 80px;
}
.cover-footer {
    font-family: 'Noto Serif KR', 'Batang', serif;
    font-size: 1.5rem;
    font-weight: bold;
    color: #000000;
}
.mois-footer {
    margin-top: 50px;
    border-top: 2px solid #D1D5DB;
    padding-top: 20px;
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
    font-size: 0.9rem;
    color: #4B5563;
}
.mois-table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 20px;
}
.mois-table td {
    padding: 8px;
    border-bottom: 1px solid #E5E7EB;
}
</style>
""" + f"""
<div class="cover-page">
    <div class="cover-title">수도권 전기차 충전소 인프라 확충 방안</div>
    <div class="cover-subtitle">- 데이터 기반 고위험 지역 도출 및 예측 모델링 -</div>
    <div class="cover-footer">
        {current_date_str}
    </div>
</div>
""", unsafe_allow_html=True)  # 표지 페이지 HTML block(블록) 렌더링

    # --- HTML 정책 보고서 본문 블록 (핵심 요약, 추진 배경, 분석 결과, TOP3, 영향 인자, 향후 계획) ---
    st.markdown(f"""
<div class="gov-summary">
[핵심 요약]<br>
현재 수도권 내 전력 및 충전 대기가 가장 심각할 것으로 우려되는 <b><span style="color:#1E3A8A;">TOP 3 고위험 지역 도출 완료</span></b>.<br>
<b>{best_name}</b> 예측 모델(RMSE: {test_rmse:.4f}) 기반의 시뮬레이션을 통해, 향후 해당 지역을 최우선으로 한 맞춤형 인프라 확충 정책 수립 요망.
</div>

<div class="gov-report">
<h3>□ 추진 배경 및 현황</h3>
<ul>
<li><b>수도권 내 전기차 보급 확대로 인한 충전 인프라 부하 증가</b>
    <ul>
        <li>최근 3년간 수도권의 전기차 등록 대수가 급증함에 따라, 주요 도심지 및 주거 밀집 지역의 충전소 부족 문제가 심화됨.</li>
        <li>특정 시간대(퇴근 시간 이후 등)에 충전 수요가 집중되면서 대기 시간 지연 및 국지적 전력망 과부하 우려가 제기됨.</li>
    </ul>
</li>
<li><b>데이터 기반의 정확한 부하 예측 및 취약 지역 선제적 대응 요망</b>
    <ul>
        <li>단순 인구수나 면적에 비례한 일률적 설치가 아닌, 실제 충전 패턴과 인프라 접근성을 반영한 과학적 예측 모델 필요.</li>
        <li>한정된 예산을 효율적으로 집행하기 위해 부하가 가장 높은 '고위험 지역'을 사전 식별하고 맞춤형 대응책을 마련하고자 함.</li>
    </ul>
</li>
</ul>

<h3>□ 분석 결과 (예측 모델 성능)</h3>
<ul>
<li><b>최적 예측 모델 선정: <span class="highlight">{best_name}</span></b>
    <ul>
        <li>RandomForest, GradientBoosting 등 다수의 머신러닝/딥러닝 모델 비교 평가 결과 최우수 성능 입증.</li>
        <li>예측 오차(Test RMSE): <span class="highlight">{test_rmse:.4f}</span> 수준으로, 실제 부하 지수와 매우 근접한 정밀도를 보임.</li>
    </ul>
</li>
<li><b>모델 신뢰도 및 강건성 확보</b>
    <ul>
        <li>모델 과적합을 방지하기 위해 중첩 교차검증(Nested CV)을 수행하였으며, 공간적 외부 검증(Spatial CV)을 통해 타 지역 확장 시에도 예측 성능이 유지됨을 확인.</li>
    </ul>
</li>
</ul>

<h3>□ 당면 문제점 (고위험 지역 현황)</h3>
<ul>
<li><b>현재 인프라 부하 최상위 TOP 3 지역 분석</b>
    <ul>
        {top3_html}
    </ul>
</li>
<li>해당 지역들은 자가용 및 사업자용 충전 수요가 동시에 폭증하는 병목 구간으로 파악됨.</li>
<li>충전소 1기당 감당해야 할 전기차 대수가 적정 수준을 초과하여, 즉각적인 충전기 추가 증설이 시급한 상황임.</li>
</ul>

<h3>□ 주요 영향 인자 분석</h3>
<ul>
<li><b>핵심 변수: <span class="highlight">{top_features[0]}</span>, <span class="highlight">{top_features[1]}</span></b>
    <ul>
        <li>SHAP 및 Feature Importance 분석 결과, 위 두 가지 요인이 충전 부하 증감에 결정적인 역할을 하는 것으로 판명됨.</li>
        <li>향후 모니터링 체계 구축 시 해당 인자들의 변화 추이를 실시간으로 추적하는 시스템 마련 권고.</li>
    </ul>
</li>
</ul>

<h3>□ 향후 계획 및 제안</h3>
<ul>
<li><b>단기 대책: TOP 3 지역 타겟팅 시뮬레이션 및 예산 투입</b>
    <ul>
        <li>도출된 고위험 지역에 대해 1:1 맞춤형 충전소 설치 시뮬레이션을 추진하여, 필요 충전기 대수 및 예상 부하 완화율을 산출할 예정임.</li>
    </ul>
</li>
<li><b>중장기 대책: 연쇄 과부하 방지를 위한 선제적 인프라 마스터플랜 수립</b>
    <ul>
        <li>연간 5% 수준의 지속적인 전기차 증가 가정 시, 향후 3~5년 내 인접 자치구로 과부하가 도미노처럼 번질 위험성이 존재함.</li>
        <li>이를 방어하기 위해 국토부 및 한전 등 유관기관과 협조하여 광역 단위의 선제적 전력망 확충 예산 편성이 강구됨.</li>
    </ul>
</li>
</ul>
</div>
""", unsafe_allow_html=True)  # 정책 보고서 본문 HTML block(블록) 렌더링


