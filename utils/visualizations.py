# ============================================================
# 파일명: visualizations.py
# 설명: 전기차 충전 인프라 시각화 모듈 — 버블맵, SHAP 설명,
#       고속도로 부하 시뮬레이션 플롯 등 다양한 시각화 기능 제공
# ============================================================

import folium  # folium(폴리움) 지도 시각화 라이브러리 import(임포트)
import numpy as np  # numpy(넘파이) 수치 연산 라이브러리를 np로 import(임포트)
import pandas as pd  # pandas(판다스) 데이터 분석 라이브러리를 pd로 import(임포트)
import streamlit as st  # streamlit(스트림릿) 웹 앱 프레임워크를 st로 import(임포트)
import streamlit.components.v1 as components  # streamlit(스트림릿) HTML component(컴포넌트) 모듈 import(임포트)
from scipy import stats  # scipy(사이파이)의 통계 모듈 stats를 import(임포트)

# --- 버블맵(Bubble Map) 생성 함수 ---
def make_bubble_map(df, metric, selected_usages, selected_region=None, scenario=None, topsis_data=None, anomalies=None):  # DataFrame(데이터프레임), 지표, 용도 목록 등을 받아 folium(폴리움) 지도 객체를 생성하는 함수 정의
    map_df = df[df["용도"].isin(selected_usages)].dropna(subset=["위도", "경도"]).copy()  # 선택된 용도에 해당하고 위도/경도가 존재하는 행만 필터링(filtering)하여 복사
    center = [37.55, 126.99] if map_df.empty else [map_df["위도"].mean(), map_df["경도"].mean()]  # 데이터가 비어 있으면 서울 중심 좌표, 아니면 평균 좌표를 지도 중심으로 설정
    m = folium.Map(location=center, zoom_start=9, tiles="CartoDB positron")  # folium(폴리움) Map(지도) 객체를 CartoDB 타일로 생성
    colors = {"자가용": "#00A699", "사업자용": "#FF5A5F"}  # 용도별 마커 색상 dictionary(딕셔너리) 정의

    # --- 1. Anomaly(이상 징후) Pulse Marker(펄스 마커) CSS 스타일 추가 ---
    if anomalies is not None and len(anomalies) > 0:  # anomalies(이상 징후) 데이터가 존재하는지 확인
        pulse_css = """
        <style>
        @keyframes pulse-anim {
            0% {
                transform: scale(0.8);
                box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7);
            }
            70% {
                transform: scale(1.2);
                box-shadow: 0 0 0 10px rgba(255, 75, 75, 0);
            }
            100% {
                transform: scale(0.8);
                box-shadow: 0 0 0 0 rgba(255, 75, 75, 0);
            }
        }
        .pulse-marker-warning {
            background-color: #FF4B4B;
            border-radius: 50%;
            height: 16px;
            width: 16px;
            animation: pulse-anim 1.5s infinite;
            border: 2px solid white;
        }
        </style>
        """  # 경고 펄스 애니메이션(animation) CSS 문자열 정의
        m.get_root().header.add_child(folium.Element(pulse_css))  # 지도 HTML header(헤더)에 CSS Element(엘리먼트)를 삽입

    max_value = max(float(map_df[metric].max()), 1.0) if not map_df.empty else 1.0  # 버블 크기 정규화를 위해 metric(지표) 최댓값 계산 (최소 1.0)

    # --- 용도별 CircleMarker(원형 마커) 레이어 추가 ---
    for usage in selected_usages:  # 선택된 용도(자가용/사업자용)를 순회(iterate)
        layer = folium.FeatureGroup(name=f"{usage} 부하도", show=True)  # 용도별 FeatureGroup(피처그룹) 레이어 생성
        usage_df = map_df[map_df["용도"] == usage]  # 해당 용도의 데이터만 필터링(filtering)
        for _, row in usage_df.iterrows():  # 각 행을 순회(iterate)하며 마커 생성
            value = float(row[metric])  # 현재 행의 metric(지표) 값을 float(실수)로 변환
            radius = 5 + 35 * np.sqrt(value / max_value)  # 값에 비례하는 버블 반지름 계산 (제곱근 스케일링)
            tooltip = (  # 마우스 hover(호버) 시 표시할 tooltip(툴팁) HTML 문자열 구성
                f"<b>{row['지역']} ({usage})</b><br>"  # 지역명과 용도를 굵게 표시
                f"{metric}: {value:,.2f}<br>"  # 선택된 metric(지표) 값 표시
                f"전기차: {row['전기차_전체대수']:,.0f}대<br>"  # 전기차 등록 대수 표시
                f"충전기: {row['전체_충전기대수']:,.0f}대<br>"  # 충전기 대수 표시
                f"총용량: {row['총용량_kW']:,.0f} kW"  # 총 충전 용량 표시
            )
            folium.CircleMarker(  # folium(폴리움) CircleMarker(원형 마커) 객체 생성
                location=[row["위도"], row["경도"]],  # 마커 위치를 위도/경도로 지정
                radius=radius,  # 계산된 반지름 적용
                color=colors.get(usage, "#4C78A8"),  # 용도에 따른 테두리 색상 지정
                weight=2,  # 테두리 두께 설정
                fill=True,  # 마커 내부 채우기 활성화
                fill_color=colors.get(usage, "#4C78A8"),  # 내부 채우기 색상 지정
                fill_opacity=0.42,  # 내부 채우기 투명도(opacity) 설정
                tooltip=tooltip,  # tooltip(툴팁) 연결
            ).add_to(layer)  # 생성된 마커를 레이어에 추가
        layer.add_to(m)  # 레이어를 지도에 추가

    # --- 2. TOPSIS Top 5 최적 입지 Gold Star(골드 스타) 마커 렌더링 ---
    if topsis_data is not None and not topsis_data.empty:  # TOPSIS 분석 결과 데이터가 존재하는지 확인
        topsis_layer = folium.FeatureGroup(name="🎯 최적 추천 입지 (MCDA)", show=True)  # TOPSIS 결과용 FeatureGroup(피처그룹) 레이어 생성
        for _, row in topsis_data.iterrows():  # TOPSIS 결과 행을 순회(iterate)
            tooltip = (  # TOPSIS 마커 tooltip(툴팁) HTML 문자열 구성
                f"<b>🎯 [최적 입지 추천] {row['지역']} ({row['용도']})</b><br>"  # 지역 및 용도 표시
                f"TOPSIS 순위: <b>{int(row['TOPSIS_순위'])}위</b><br>"  # TOPSIS 순위 표시
                f"TOPSIS 점수: {row['TOPSIS_점수']:.4f}<br>"  # TOPSIS 점수 표시
                f"인프라 부하: {row['인프라_부하지수']:.2f}<br>"  # 인프라 부하지수 표시
                f"전력 부하: {row['전력_부하지수']:.2f}"  # 전력 부하지수 표시
            )
            folium.Marker(  # folium(폴리움) Marker(마커) 객체 생성 (별 아이콘)
                location=[row["위도"], row["경도"]],  # 마커 위치를 위도/경도로 지정
                tooltip=tooltip,  # tooltip(툴팁) 연결
                icon=folium.Icon(color="orange", icon="star", prefix="fa"),  # FontAwesome(폰트어썸) 별 아이콘을 주황색으로 설정
            ).add_to(topsis_layer)  # 마커를 TOPSIS 레이어에 추가
        topsis_layer.add_to(m)  # TOPSIS 레이어를 지도에 추가

    # --- 3. Anomaly Detection(이상 탐지) Pulse Marker(펄스 마커) 렌더링 ---
    if anomalies is not None:  # anomalies(이상 징후) 데이터가 None이 아닌지 확인
        anomaly_layer = folium.FeatureGroup(name="🚨 이상 징후 관제 (Anomaly)", show=True)  # 이상 징후 표시용 FeatureGroup(피처그룹) 레이어 생성
        items = anomalies.to_dict(orient="records") if isinstance(anomalies, pd.DataFrame) else anomalies  # DataFrame(데이터프레임)이면 dict(딕셔너리) 리스트로 변환
        for item in items:  # 각 이상 징후 항목을 순회(iterate)
            tooltip = (  # 이상 징후 마커 tooltip(툴팁) HTML 문자열 구성
                f"<b>🚨 [이상 징후 포착] {item['지역']} ({item['용도']})</b><br>"  # 지역 및 용도 표시
                f"상태: <span style='color:red; font-weight:bold;'>위험 (Warning)</span><br>"  # 위험 상태를 빨간색으로 표시
                f"이상 유형: {item.get('anomaly_type', '기기 이상')}<br>"  # 이상 유형 표시 (기본값: 기기 이상)
                f"커넥터 온도: {item.get('temperature', 0.0):.1f}°C<br>"  # 커넥터 온도 표시
                f"전압 변동성: {item.get('voltage_std', 0.0):.2f} V<br>"  # 전압 표준편차 표시
                f"패킷 손실률: {item.get('packet_loss', 0.0):.1f}%"  # 패킷 손실률 표시
            )
            folium.Marker(  # folium(폴리움) Marker(마커) 객체 생성 (펄스 DivIcon)
                location=[item["위도"], item["경도"]],  # 마커 위치를 위도/경도로 지정
                tooltip=tooltip,  # tooltip(툴팁) 연결
                icon=folium.DivIcon(  # DivIcon(HTML 기반 아이콘) 사용
                    icon_size=(16, 16),  # 아이콘 크기를 16x16 pixel(픽셀)로 설정
                    icon_anchor=(8, 8),  # 아이콘 앵커(anchor) 포인트를 중앙으로 설정
                    html='<div class="pulse-marker-warning"></div>'  # 펄스 애니메이션(animation) CSS 클래스 적용
                )
            ).add_to(anomaly_layer)  # 마커를 이상 징후 레이어에 추가
        anomaly_layer.add_to(m)  # 이상 징후 레이어를 지도에 추가

    # --- 4. Scenario(시나리오) 개입 시뮬레이션 마커 ---
    if selected_region and scenario:  # 선택된 지역과 scenario(시나리오)가 모두 존재하는지 확인
        target = map_df[map_df["지역"] == selected_region].head(1)  # 선택된 지역의 첫 번째 행 추출
        if not target.empty:  # 해당 지역 데이터가 존재하면 실행
            row = target.iloc[0]  # 첫 번째 행을 Series(시리즈)로 가져오기
            folium.Marker(  # 신규 설치 시뮬레이션 Marker(마커) 생성
                [row["위도"], row["경도"]],  # 마커 위치를 위도/경도로 지정
                tooltip=(  # tooltip(툴팁) HTML 문자열 구성
                    f"{selected_region} 신규 설치 시뮬레이션<br>"  # 시뮬레이션 대상 지역명 표시
                    f"추가 용량: {scenario['added_kw']:,.0f} kW<br>"  # 추가될 충전 용량 표시
                    f"부하 감소: {scenario['reduction_pct']:.1f}%"  # 예상 부하 감소율 표시
                ),
                icon=folium.Icon(color="blue", icon="plus-sign"),  # 파란색 플러스 아이콘 사용
            ).add_to(m)  # 마커를 지도에 직접 추가

    folium.LayerControl(collapsed=False).add_to(m)  # 레이어 컨트롤(Layer Control) 패널을 지도에 추가 (펼쳐진 상태)
    return m  # 완성된 folium(폴리움) Map(지도) 객체 반환(리턴)

# --- TableOne(테이블원) 통계 요약 테이블 생성 함수 ---
def make_tableone(final_data, table_cols):  # 최종 데이터와 테이블 컬럼(column) 목록을 받아 통계 요약 테이블을 생성하는 함수 정의
    try:  # TableOne(테이블원) 라이브러리 사용 시도
        from tableone import TableOne  # tableone(테이블원) 라이브러리 import(임포트)

        table_data = final_data[["용도", *table_cols]].rename(columns={"용도": "usage"})  # 용도 컬럼명을 "usage"로 rename(이름 변경)하여 테이블 데이터 준비
        table = TableOne(table_data, columns=table_cols, groupby="usage", pval=True)  # 용도별 그룹으로 TableOne(테이블원) 객체 생성 (p-value 포함)
        return "tableone", table.tabulate(tablefmt="github")  # GitHub(깃허브) markdown(마크다운) 형식으로 변환하여 반환(리턴)
    except Exception:  # TableOne(테이블원) 라이브러리가 없거나 오류 발생 시
        rows = []  # fallback(대체) 방식을 위한 결과 리스트 초기화
        for col in table_cols:  # 각 테이블 컬럼을 순회(iterate)
            private = final_data[final_data["용도"] == "자가용"][col]  # 자가용 데이터 추출
            business = final_data[final_data["용도"] == "사업자용"][col]  # 사업자용 데이터 추출
            _, p_val = stats.ttest_ind(private, business, equal_var=False, nan_policy="omit")  # Welch t-test(웰치 t-검정)로 p-value 산출
            rows.append(  # 결과 행을 리스트에 추가
                {
                    "Variable": col,  # 변수명 저장
                    "자가용 mean±sd": f"{private.mean():,.2f} ± {private.std():,.2f}",  # 자가용 평균±표준편차 문자열 생성
                    "사업자용 mean±sd": f"{business.mean():,.2f} ± {business.std():,.2f}",  # 사업자용 평균±표준편차 문자열 생성
                    "p-value": p_val,  # p-value 값 저장
                }
            )
        return "fallback", pd.DataFrame(rows)  # fallback(대체) 방식의 DataFrame(데이터프레임) 결과 반환(리턴)

# --- SHAP(샵) 값 캐싱(caching) 함수 ---
@st.cache_data(show_spinner="SHAP 기여도 분석 및 요약 플롯 생성 중...")  # Streamlit(스트림릿) cache(캐시) decorator(데코레이터) — SHAP 값 캐싱
def get_cached_shap_values(best_name, _model, X_sample):  # 최적 모델명, 모델 객체, 샘플 데이터를 받아 SHAP 값을 계산하는 함수 정의
    import shap  # shap(샵) 모델 설명 라이브러리 import(임포트)
    explainer = shap.Explainer(_model.predict, X_sample)  # SHAP Explainer(설명자) 객체를 모델 predict(예측) 함수와 배경 데이터로 생성
    shap_values = explainer(X_sample)  # 샘플 데이터에 대한 SHAP 값 계산
    return shap_values  # 계산된 SHAP 값 객체 반환(리턴)

# --- 지역별 Local SHAP/LIME Force Plot(포스 플롯) 값 캐싱(caching) 함수 ---
@st.cache_data(show_spinner="지역별 LIME/Force plot 값 계산 중...")  # Streamlit(스트림릿) cache(캐시) decorator(데코레이터) — Local SHAP 값 캐싱
def get_cached_local_shap(best_name, _model, X_sample, local_x):  # 모델명, 모델, 샘플, 개별 관측치를 받아 Local SHAP 값을 계산하는 함수 정의
    import shap  # shap(샵) 모델 설명 라이브러리 import(임포트)
    explainer = shap.Explainer(_model.predict, X_sample)  # SHAP Explainer(설명자) 객체 생성
    local_exp = explainer(local_x)  # 개별 관측치에 대한 SHAP 설명 계산
    try:  # expected_value(기대값) 접근 시도
        base_val = explainer.expected_value  # Explainer(설명자)의 expected_value(기대값) 추출
    except AttributeError:  # expected_value attribute(속성)가 없는 경우
        base_val = local_exp.base_values[0] if hasattr(local_exp, 'base_values') else 0  # base_values에서 가져오거나 0으로 설정
    if isinstance(base_val, (list, np.ndarray)):  # base_val이 list(리스트)나 array(배열)인지 확인
        base_val = base_val[0]  # 첫 번째 원소만 추출
    vals = local_exp.values[0] if hasattr(local_exp, 'values') else local_exp[0]  # SHAP 기여도 값 배열 추출
    return base_val, vals  # 기저(base) 값과 SHAP 기여도 값을 tuple(튜플)로 반환(리턴)

# --- SHAP Summary Plot(요약 플롯) 또는 Fallback(대체) 설명 렌더링 함수 ---
def render_shap_or_fallback(model_state, selected_feature, local_x=None):  # 모델 상태, 선택된 피처(feature), 개별 관측치를 받아 SHAP 시각화를 렌더링하는 함수 정의
    best_model = model_state["models"][model_state["best_name"]]  # 최적 모델 객체 추출
    X_all = model_state["X"]  # 전체 feature(피처) 데이터 추출
    try:  # SHAP 시각화 시도
        import matplotlib.pyplot as plt  # matplotlib(맷플롯립) pyplot 모듈 import(임포트)
        import shap  # shap(샵) 모델 설명 라이브러리 import(임포트)

        sample = X_all.sample(min(60, len(X_all)), random_state=42)  # 최대 60개 샘플을 랜덤(random) 추출 (재현성을 위해 seed(시드) 42 사용)
        shap_values = get_cached_shap_values(model_state["best_name"], best_model, sample)  # 캐싱(caching)된 SHAP 값 가져오기

        st.success("SHAP 패키지를 사용해 summary plot을 생성했습니다.")  # 성공 메시지를 Streamlit(스트림릿) UI에 표시
        fig = plt.figure(figsize=(7, 4.5))  # matplotlib(맷플롯립) Figure(피규어) 객체 생성 (7x4.5 인치)
        shap.summary_plot(shap_values, sample, show=False, max_display=10)  # SHAP summary plot(요약 플롯) 생성 (상위 10개 feature 표시, 자동 표시 비활성화)
        st.pyplot(fig, clear_figure=True, use_container_width=False)  # Streamlit(스트림릿)에 matplotlib(맷플롯립) 그래프 렌더링

        # --- Local SHAP Force Plot(포스 플롯) 렌더링 ---
        if local_x is not None and len(local_x) > 0:  # 개별 관측치 데이터가 존재하는지 확인
            base_val, vals = get_cached_local_shap(model_state["best_name"], best_model, sample, local_x)  # 캐싱(caching)된 Local SHAP 값 가져오기

            force = shap.force_plot(  # SHAP force_plot(포스 플롯) 객체 생성
                base_val,  # 기저(base) 예측값
                vals,  # 개별 SHAP 기여도 값
                local_x.iloc[0],  # 개별 관측치의 feature(피처) 값
                matplotlib=False,  # JavaScript(자바스크립트) 기반 interactive(인터랙티브) 렌더링 사용
            )

            try:  # SHAP JavaScript(자바스크립트) 코드 가져오기 시도
                js_code = shap.getjs()  # SHAP JavaScript(자바스크립트) 번들 코드 추출
            except AttributeError:  # getjs() 메서드가 없는 경우 (버전 호환성 이슈)
                js_code = "<script>window.shap = window.shap || {};</script>"  # fallback(대체) JavaScript(자바스크립트) 코드 사용

            components.html(js_code + force.html(), height=140)  # Streamlit(스트림릿) HTML component(컴포넌트)로 force plot 렌더링
        return True  # SHAP 시각화 성공 시 True 반환(리턴)
    except Exception as exc:  # SHAP 관련 예외(exception) 발생 시
        import traceback  # traceback(트레이스백) 모듈 import(임포트)
        traceback.print_exc()  # 전체 traceback(트레이스백)을 콘솔에 출력
        st.info(f"SHAP 패키지가 호환되지 않거나 실행 중 에러가 발생했습니다. 대체 설명을 표시합니다. 사유: {type(exc).__name__} - {str(exc)}")  # 오류 메시지를 Streamlit(스트림릿) info로 표시
        return False  # SHAP 시각화 실패 시 False 반환(리턴)

import plotly.graph_objects as go  # plotly(플로틀리) graph_objects 모듈을 go로 import(임포트)

# --- 고속도로 휴게소 부하 시뮬레이션 Edge Plot(엣지 플롯) 렌더링 함수 ---
def render_highway_edge_plot(hw_df, scenario):  # 고속도로 DataFrame(데이터프레임)과 scenario(시나리오)를 받아 Mapbox(맵박스) 플롯을 생성하는 함수 정의
    fig = go.Figure()  # plotly(플로틀리) Figure(피규어) 객체 생성
    routes = hw_df["routeName"].unique()  # 고유한 노선명 배열 추출
    OFFSET_DEG = 0.008  # 상행/하행 분리 표시를 위한 경도 offset(오프셋) 값 (약 0.008도)

    # --- 노선별 상행/하행 트레이스(trace) 추가 ---
    for route in routes:  # 각 노선을 순회(iterate)
        route_df = hw_df[hw_df["routeName"] == route].copy()  # 해당 노선 데이터 필터링(filtering) 및 복사
        route_df = route_df.sort_values("위도")  # 위도 기준 정렬(sort)

        up_df = route_df[route_df["unitName"].str.endswith("상")].copy()  # 상행선 데이터 필터링(filtering)
        down_df = route_df[route_df["unitName"].str.endswith("하")].copy()  # 하행선 데이터 필터링(filtering)
        other_df = route_df[~route_df["unitName"].str.endswith(("상", "하"))].copy()  # 상/하행 구분이 없는 데이터 필터링(filtering)

        def add_trace(df, is_upbound, is_downbound):  # DataFrame(데이터프레임)과 방향 flag(플래그)를 받아 trace(트레이스)를 추가하는 내부 함수 정의
            if df.empty: return  # 데이터가 비어 있으면 즉시 반환(리턴)

            lon_offset = 0.0  # 경도 offset(오프셋) 초기값 설정
            if is_upbound: lon_offset = OFFSET_DEG  # 상행이면 양의 offset(오프셋) 적용
            elif is_downbound: lon_offset = -OFFSET_DEG  # 하행이면 음의 offset(오프셋) 적용

            lons = df["경도"] + lon_offset  # 경도에 offset(오프셋) 적용
            lats = df["위도"]  # 위도 값 추출

            colors = []  # 부하 점수별 색상 리스트 초기화
            for score in df["부하_예측점수"]:  # 각 부하 예측 점수를 순회(iterate)
                if score >= 80: colors.append("red")  # 80점 이상이면 빨간색 (고부하)
                elif score >= 60: colors.append("orange")  # 60점 이상이면 주황색 (중부하)
                else: colors.append("green")  # 60점 미만이면 녹색 (저부하)

            fig.add_trace(go.Scattermapbox(  # 노선 연결 line(선) trace(트레이스) 추가
                mode="lines", lon=lons, lat=lats,  # 선 모드로 경도/위도 설정
                line=dict(width=2, color="gray"),  # 회색 2px 선 스타일
                hoverinfo="none", showlegend=False  # hover(호버) 정보 및 범례 비활성화
            ))

            direction = "상행" if is_upbound else "하행" if is_downbound else "양방향"  # 방향 라벨(label) 결정
            hover_text = df["unitName"] + "<br>총용량: " + df["총용량_kW"].astype(str) + "kW<br>부하: " + df["부하_예측점수"].round(1).astype(str)  # hover(호버) 텍스트 구성
            fig.add_trace(go.Scattermapbox(  # 휴게소 포인트 marker(마커) trace(트레이스) 추가
                mode="markers", lon=lons, lat=lats,  # marker(마커) 모드로 경도/위도 설정
                marker=dict(size=8, color=colors),  # 부하별 색상 및 크기 설정
                text=hover_text, hoverinfo="text",  # hover(호버) 텍스트 연결
                name=f"{route} {direction}"  # trace(트레이스) 이름을 노선명 + 방향으로 설정
            ))

        add_trace(up_df, True, False)  # 상행선 trace(트레이스) 추가
        add_trace(down_df, False, True)  # 하행선 trace(트레이스) 추가
        add_trace(other_df, False, False)  # 양방향(기타) trace(트레이스) 추가

    fig.update_layout(  # plotly(플로틀리) 레이아웃 업데이트
        mapbox=dict(  # Mapbox(맵박스) 설정
            style="carto-positron",  # CartoDB Positron 타일 스타일 사용
            center=dict(lat=hw_df["위도"].mean(), lon=hw_df["경도"].mean()),  # 지도 중심을 데이터 평균 좌표로 설정
            zoom=6.5  # 초기 줌(zoom) 레벨 설정
        ),
        margin=dict(l=0, r=0, t=30, b=0),  # 레이아웃 margin(여백) 설정
        title=f"{scenario} 시뮬레이션 결과 (상하행선 분리)",  # 차트 제목 설정
        showlegend=False  # 범례(legend) 비활성화
    )
    return fig  # 완성된 plotly(플로틀리) Figure(피규어) 객체 반환(리턴)
