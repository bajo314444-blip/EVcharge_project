import folium
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from scipy import stats

def make_bubble_map(df, metric, selected_usages, selected_region=None, scenario=None, topsis_data=None, anomalies=None):
    map_df = df[df["용도"].isin(selected_usages)].dropna(subset=["위도", "경도"]).copy()
    center = [37.55, 126.99] if map_df.empty else [map_df["위도"].mean(), map_df["경도"].mean()]
    m = folium.Map(location=center, zoom_start=9, tiles="CartoDB positron")
    colors = {"자가용": "#00A699", "사업자용": "#FF5A5F"}
    
    # 1. Add Pulse Marker CSS to header if anomalies are present
    if anomalies is not None and len(anomalies) > 0:
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
        """
        m.get_root().header.add_child(folium.Element(pulse_css))

    max_value = max(float(map_df[metric].max()), 1.0) if not map_df.empty else 1.0

    for usage in selected_usages:
        layer = folium.FeatureGroup(name=f"{usage} 부하도", show=True)
        usage_df = map_df[map_df["용도"] == usage]
        for _, row in usage_df.iterrows():
            value = float(row[metric])
            radius = 5 + 35 * np.sqrt(value / max_value)
            tooltip = (
                f"<b>{row['지역']} ({usage})</b><br>"
                f"{metric}: {value:,.2f}<br>"
                f"전기차: {row['전기차_전체대수']:,.0f}대<br>"
                f"충전기: {row['전체_충전기대수']:,.0f}대<br>"
                f"총용량: {row['총용량_kW']:,.0f} kW"
            )
            folium.CircleMarker(
                location=[row["위도"], row["경도"]],
                radius=radius,
                color=colors.get(usage, "#4C78A8"),
                weight=2,
                fill=True,
                fill_color=colors.get(usage, "#4C78A8"),
                fill_opacity=0.42,
                tooltip=tooltip,
            ).add_to(layer)
        layer.add_to(m)

    # 2. Render TOPSIS Top 5 Gold Star Markers
    if topsis_data is not None and not topsis_data.empty:
        topsis_layer = folium.FeatureGroup(name="🎯 최적 추천 입지 (MCDA)", show=True)
        for _, row in topsis_data.iterrows():
            tooltip = (
                f"<b>🎯 [최적 입지 추천] {row['지역']} ({row['용도']})</b><br>"
                f"TOPSIS 순위: <b>{int(row['TOPSIS_순위'])}위</b><br>"
                f"TOPSIS 점수: {row['TOPSIS_점수']:.4f}<br>"
                f"인프라 부하: {row['인프라_부하지수']:.2f}<br>"
                f"전력 부하: {row['전력_부하지수']:.2f}"
            )
            folium.Marker(
                location=[row["위도"], row["경도"]],
                tooltip=tooltip,
                icon=folium.Icon(color="orange", icon="star", prefix="fa"),
            ).add_to(topsis_layer)
        topsis_layer.add_to(m)

    # 3. Render Anomaly Detection Pulse Markers
    if anomalies is not None:
        anomaly_layer = folium.FeatureGroup(name="🚨 이상 징후 관제 (Anomaly)", show=True)
        items = anomalies.to_dict(orient="records") if isinstance(anomalies, pd.DataFrame) else anomalies
        for item in items:
            tooltip = (
                f"<b>🚨 [이상 징후 포착] {item['지역']} ({item['용도']})</b><br>"
                f"상태: <span style='color:red; font-weight:bold;'>위험 (Warning)</span><br>"
                f"이상 유형: {item.get('anomaly_type', '기기 이상')}<br>"
                f"커넥터 온도: {item.get('temperature', 0.0):.1f}°C<br>"
                f"전압 변동성: {item.get('voltage_std', 0.0):.2f} V<br>"
                f"패킷 손실률: {item.get('packet_loss', 0.0):.1f}%"
            )
            folium.Marker(
                location=[item["위도"], item["경도"]],
                tooltip=tooltip,
                icon=folium.DivIcon(
                    icon_size=(16, 16),
                    icon_anchor=(8, 8),
                    html='<div class="pulse-marker-warning"></div>'
                )
            ).add_to(anomaly_layer)
        anomaly_layer.add_to(m)

    # 4. Scenario intervention marker
    if selected_region and scenario:
        target = map_df[map_df["지역"] == selected_region].head(1)
        if not target.empty:
            row = target.iloc[0]
            folium.Marker(
                [row["위도"], row["경도"]],
                tooltip=(
                    f"{selected_region} 신규 설치 시뮬레이션<br>"
                    f"추가 용량: {scenario['added_kw']:,.0f} kW<br>"
                    f"부하 감소: {scenario['reduction_pct']:.1f}%"
                ),
                icon=folium.Icon(color="blue", icon="plus-sign"),
            ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m

def make_tableone(final_data, table_cols):
    try:
        from tableone import TableOne

        table_data = final_data[["용도", *table_cols]].rename(columns={"용도": "usage"})
        table = TableOne(table_data, columns=table_cols, groupby="usage", pval=True)
        return "tableone", table.tabulate(tablefmt="github")
    except Exception:
        rows = []
        for col in table_cols:
            private = final_data[final_data["용도"] == "자가용"][col]
            business = final_data[final_data["용도"] == "사업자용"][col]
            _, p_val = stats.ttest_ind(private, business, equal_var=False, nan_policy="omit")
            rows.append(
                {
                    "Variable": col,
                    "자가용 mean±sd": f"{private.mean():,.2f} ± {private.std():,.2f}",
                    "사업자용 mean±sd": f"{business.mean():,.2f} ± {business.std():,.2f}",
                    "p-value": p_val,
                }
            )
        return "fallback", pd.DataFrame(rows)

@st.cache_data(show_spinner="SHAP 기여도 분석 및 요약 플롯 생성 중...")
def get_cached_shap_values(best_name, _model, X_sample):
    import shap
    explainer = shap.Explainer(_model.predict, X_sample)
    shap_values = explainer(X_sample)
    return shap_values

@st.cache_data(show_spinner="지역별 LIME/Force plot 값 계산 중...")
def get_cached_local_shap(best_name, _model, X_sample, local_x):
    import shap
    explainer = shap.Explainer(_model.predict, X_sample)
    local_exp = explainer(local_x)
    try:
        base_val = explainer.expected_value
    except AttributeError:
        base_val = local_exp.base_values[0] if hasattr(local_exp, 'base_values') else 0
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[0]
    vals = local_exp.values[0] if hasattr(local_exp, 'values') else local_exp[0]
    return base_val, vals

def render_shap_or_fallback(model_state, selected_feature, local_x=None):
    best_model = model_state["models"][model_state["best_name"]]
    X_all = model_state["X"]
    try:
        import matplotlib.pyplot as plt
        import shap

        sample = X_all.sample(min(60, len(X_all)), random_state=42)
        shap_values = get_cached_shap_values(model_state["best_name"], best_model, sample)

        st.success("SHAP 패키지를 사용해 summary plot을 생성했습니다.")
        fig = plt.figure(figsize=(7, 4.5))
        shap.summary_plot(shap_values, sample, show=False, max_display=10)
        st.pyplot(fig, clear_figure=True, use_container_width=False)

        if local_x is not None and len(local_x) > 0:
            base_val, vals = get_cached_local_shap(model_state["best_name"], best_model, sample, local_x)
            
            force = shap.force_plot(
                base_val,
                vals,
                local_x.iloc[0],
                matplotlib=False,
            )
            
            try:
                js_code = shap.getjs()
            except AttributeError:
                js_code = "<script>window.shap = window.shap || {};</script>"
                
            components.html(js_code + force.html(), height=140)
        return True
    except Exception as exc:
        import traceback
        traceback.print_exc()
        st.info(f"SHAP 패키지가 호환되지 않거나 실행 중 에러가 발생했습니다. 대체 설명을 표시합니다. 사유: {type(exc).__name__} - {str(exc)}")
        return False

import plotly.graph_objects as go

def render_highway_edge_plot(hw_df, scenario):
    fig = go.Figure()
    routes = hw_df["routeName"].unique()
    OFFSET_DEG = 0.008 
    
    for route in routes:
        route_df = hw_df[hw_df["routeName"] == route].copy()
        route_df = route_df.sort_values("위도")
        
        up_df = route_df[route_df["unitName"].str.endswith("상")].copy()
        down_df = route_df[route_df["unitName"].str.endswith("하")].copy()
        other_df = route_df[~route_df["unitName"].str.endswith(("상", "하"))].copy()
        
        def add_trace(df, is_upbound, is_downbound):
            if df.empty: return
            
            lon_offset = 0.0
            if is_upbound: lon_offset = OFFSET_DEG
            elif is_downbound: lon_offset = -OFFSET_DEG
                
            lons = df["경도"] + lon_offset
            lats = df["위도"]
            
            colors = []
            for score in df["부하_예측점수"]:
                if score >= 80: colors.append("red")
                elif score >= 60: colors.append("orange")
                else: colors.append("green")
                
            fig.add_trace(go.Scattermapbox(
                mode="lines", lon=lons, lat=lats,
                line=dict(width=2, color="gray"),
                hoverinfo="none", showlegend=False
            ))
            
            direction = "상행" if is_upbound else "하행" if is_downbound else "양방향"
            hover_text = df["unitName"] + "<br>총용량: " + df["총용량_kW"].astype(str) + "kW<br>부하: " + df["부하_예측점수"].round(1).astype(str)
            fig.add_trace(go.Scattermapbox(
                mode="markers", lon=lons, lat=lats,
                marker=dict(size=8, color=colors),
                text=hover_text, hoverinfo="text",
                name=f"{route} {direction}"
            ))

        add_trace(up_df, True, False)
        add_trace(down_df, False, True)
        add_trace(other_df, False, False)
        
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=hw_df["위도"].mean(), lon=hw_df["경도"].mean()),
            zoom=6.5
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"{scenario} 시뮬레이션 결과 (상하행선 분리)",
        showlegend=False
    )
    return fig
