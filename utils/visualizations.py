import folium
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from scipy import stats

def make_bubble_map(df, metric, selected_usages, selected_region=None, scenario=None):
    map_df = df[df["용도"].isin(selected_usages)].dropna(subset=["위도", "경도"]).copy()
    center = [37.55, 126.99] if map_df.empty else [map_df["위도"].mean(), map_df["경도"].mean()]
    m = folium.Map(location=center, zoom_start=9, tiles="CartoDB positron")
    colors = {"자가용": "#00A699", "사업자용": "#FF5A5F"}
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

def render_shap_or_fallback(model_state, selected_feature, local_x=None):
    best_model = model_state["models"][model_state["best_name"]]
    X_all = model_state["X"]
    try:
        import matplotlib.pyplot as plt
        import shap

        sample = X_all.sample(min(60, len(X_all)), random_state=42)
        explainer = shap.Explainer(best_model.predict, sample)
        shap_values = explainer(sample)

        st.success("SHAP 패키지를 사용해 summary plot과 dependence plot을 생성했습니다.")
        fig = plt.figure(figsize=(9, 5))
        shap.summary_plot(shap_values, sample, show=False, max_display=10)
        st.pyplot(fig, clear_figure=True)

        fig = plt.figure(figsize=(8, 5))
        shap.plots.scatter(shap_values[:, selected_feature], show=False)
        st.pyplot(fig, clear_figure=True)

        if local_x is not None and len(local_x) > 0:
            local_exp = explainer(local_x)
            force = shap.force_plot(
                explainer.expected_value,
                local_exp.values[0],
                local_x.iloc[0],
                matplotlib=False,
            )
            components.html(shap.getjs() + force.html(), height=180)
        return True
    except Exception as exc:
        st.info(f"SHAP 패키지가 없거나 실행할 수 없어 대체 설명을 표시합니다. 사유: {type(exc).__name__}")
        return False
