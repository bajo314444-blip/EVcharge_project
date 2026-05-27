import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from streamlit_folium import st_folium

from utils.data_processing import DEFAULT_DATA_DIR, METRO_SHORT, load_all_data
from utils.models import make_feature_matrix, train_models
from utils.visualizations import make_bubble_map, make_tableone, render_shap_or_fallback

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="수도권 전기차 충전소 부하 예측",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("수도권 전기차 충전소 부하 예측 웹서비스")
st.caption("자가용과 사업자용 수요를 비교하고, 신규 충전소 설치 시 부하 완화 효과를 지도에서 확인합니다.")

with st.sidebar:
    st.header("데이터 설정")
    data_dir = st.text_input("CSV/Excel 데이터 폴더", value=str(DEFAULT_DATA_DIR))
    st.caption("폴더 안의 원본 파일명을 그대로 사용합니다.")

try:
    final_data, monthly_data, hourly_data = load_all_data(data_dir)
except Exception as exc:
    st.error(f"데이터를 불러오지 못했습니다: {exc}")
    st.stop()

model_state = train_models(final_data.to_json(orient="split"))

with st.sidebar:
    st.header("지도 필터")
    usage_options = st.multiselect("용도", ["자가용", "사업자용"], default=["자가용", "사업자용"])
    metric = st.selectbox("버블 크기 기준", ["전력_부하지수", "인프라_부하지수", "총_전력판매량"])
    province_filter = st.multiselect("시도", METRO_SHORT, default=METRO_SHORT)

filtered = final_data[final_data["시도"].isin(province_filter)].copy()
if usage_options:
    filtered = filtered[filtered["용도"].isin(usage_options)].copy()

top_region = filtered.sort_values("전력_부하지수", ascending=False).head(1)
col1, col2, col3, col4 = st.columns(4)
col1.metric("분석 지역-용도 행", f"{len(filtered):,}개")
col2.metric("최고 부하 지역", top_region["지역"].iloc[0] if not top_region.empty else "-")
col3.metric("최고 전력 부하지수", f"{top_region['전력_부하지수'].iloc[0]:,.2f}" if not top_region.empty else "-")
col4.metric("학습 기준 최고 모델", model_state["best_name"])

tabs = st.tabs(
    [
        "지도 버블맵",
        "월별 부하 변화",
        "설치 시뮬레이션",
        "예측 모델 비교",
        "통계/군집 분석",
        "SHAP/LIME 설명",
        "조건 충족표",
    ]
)

with tabs[0]:
    st.subheader("현재 부하 버블맵")
    st.caption("지도 레이어에서 자가용과 사업자용을 켜고 끌 수 있습니다.")
    left, right = st.columns([1.35, 0.9])
    with left:
        st_folium(make_bubble_map(filtered, metric, usage_options), height=650, use_container_width=True)
    with right:
        st.markdown("#### 고부하 상위 지역")
        top_table = (
            filtered.sort_values("전력_부하지수", ascending=False)
            [["지역", "용도", "전기차_전체대수", "전체_충전기대수", "총용량_kW", "전력_부하지수", "인프라_부하지수"]]
            .head(15)
        )
        st.dataframe(top_table, use_container_width=True, hide_index=True)
        fig = px.bar(
            top_table.sort_values("전력_부하지수"),
            x="전력_부하지수",
            y="지역",
            color="용도",
            orientation="h",
            color_discrete_map={"자가용": "#00A699", "사업자용": "#FF5A5F"},
        )
        st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("환경부 공공급속 충전기 연월별 부하 변화")
    st.caption("새로 추가한 환경부 2017-2025년 월별 파일을 전처리 단계부터 반영했습니다.")
    available_regions = sorted(monthly_data["지역"].dropna().unique())
    default_monthly_regions = (
        monthly_data.groupby("지역")["월_충전량"].sum().sort_values(ascending=False).head(5).index.tolist()
    )
    selected_monthly_regions = st.multiselect("월별 추이를 볼 지역", available_regions, default=default_monthly_regions)
    monthly_metric = st.selectbox("월별 지표", ["월별_부하지수", "월_충전량", "월_충전횟수", "월_충전시간", "운영_충전소수"])
    month_view = monthly_data[monthly_data["지역"].isin(selected_monthly_regions)].copy()
    fig = px.line(month_view, x="연월", y=monthly_metric, color="지역", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    latest_month = monthly_data["연월"].max()
    latest = monthly_data[monthly_data["연월"] == latest_month].copy()
    st.markdown(f"#### 최신 월 기준 고부하 지역: {latest_month.strftime('%Y-%m')}")
    st.dataframe(
        latest.sort_values("월별_부하지수", ascending=False)
        [["지역", "월_충전량", "월_충전횟수", "월_충전시간", "운영_충전소수", "월별_부하지수"]]
        .head(20),
        use_container_width=True,
        hide_index=True,
    )

with tabs[2]:
    st.subheader("신규 충전소 설치 효과 시뮬레이션")
    st.caption("선택 지역에 충전기를 추가했을 때 전력 부하지수가 얼마나 낮아지는지 계산합니다.")
    sim_col1, sim_col2 = st.columns([0.9, 1.2])
    with sim_col1:
        region_list = sorted(final_data["지역"].unique())
        default_idx = region_list.index(top_region["지역"].iloc[0]) if not top_region.empty else 0
        sim_region = st.selectbox("설치 후보 지역", region_list, index=default_idx)
        sim_usage = st.radio("계산 기준 용도", ["자가용", "사업자용"], horizontal=True)
        charger_count = st.slider("추가 충전기 수", 1, 80, 10)
        charger_kw = st.select_slider("충전기 1대 용량(kW)", options=[50, 100, 150, 200, 350], value=100)
        target_row = final_data[(final_data["지역"] == sim_region) & (final_data["용도"] == sim_usage)].iloc[0]
        added_kw = charger_count * charger_kw
        before = float(target_row["전력_부하지수"])
        after = float(target_row["총_전력판매량"] / (target_row["총용량_kW"] + added_kw))
        reduction_pct = (before - after) / before * 100 if before else 0
        st.metric("현재 전력 부하지수", f"{before:,.2f}")
        st.metric("설치 후 예상 부하지수", f"{after:,.2f}", delta=f"-{reduction_pct:.1f}%")
        st.metric("추가 공급 용량", f"{added_kw:,.0f} kW")
    with sim_col2:
        scenario_df = pd.DataFrame(
            {
                "상태": ["현재", "설치 후"],
                "전력 부하지수": [before, after],
                "총 공급용량(kW)": [target_row["총용량_kW"], target_row["총용량_kW"] + added_kw],
            }
        )
        fig = px.bar(scenario_df, x="상태", y="전력 부하지수", color="상태", text_auto=".2f")
        st.plotly_chart(fig, use_container_width=True)
        st_folium(
            make_bubble_map(
                final_data[final_data["지역"].isin([sim_region])],
                "전력_부하지수",
                [sim_usage],
                selected_region=sim_region,
                scenario={"added_kw": added_kw, "reduction_pct": reduction_pct},
            ),
            height=360,
            use_container_width=True,
        )

with tabs[3]:
    st.subheader("예측 모델 성능 비교")
    st.caption("머신러닝 5개와 딥러닝 2개(CNN, Transformer 계열)를 같은 데이터 분할로 비교합니다.")
    metrics = model_state["metrics"]
    st.dataframe(metrics.round(4), use_container_width=True, hide_index=True)

    test_metrics = metrics[metrics["Split"] == "Test"].sort_values("RMSE")
    fig = px.bar(
        test_metrics,
        x="Model",
        y="RMSE",
        color="Group",
        title="Test RMSE 비교",
        text_auto=".1f",
    )
    st.plotly_chart(fig, use_container_width=True)

    pred = model_state["predictions"]
    best_pred = pred[pred["Model"] == model_state["best_name"]].copy()
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(best_pred, x="Actual", y="Predicted", color="용도", hover_name="지역", title="Actual vs Prediction")
        min_v = min(best_pred["Actual"].min(), best_pred["Predicted"].min())
        max_v = max(best_pred["Actual"].max(), best_pred["Predicted"].max())
        fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", name="완전 일치"))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        best_pred["Residual"] = best_pred["Actual"] - best_pred["Predicted"]
        fig = px.scatter(best_pred, x="Predicted", y="Residual", color="용도", hover_name="지역", title="Residual Plot")
        fig.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

    qq = stats.probplot(best_pred["Residual"], dist="norm")
    qq_df = pd.DataFrame({"Theoretical": qq[0][0], "Ordered residual": qq[0][1]})
    fig = px.scatter(qq_df, x="Theoretical", y="Ordered residual", title="QQ Plot")
    st.plotly_chart(fig, use_container_width=True)

    threshold = model_state["y"].quantile(0.7)
    roc_rows = []
    roc_fig = go.Figure()
    y_test_binary = (model_state["y_test"] >= threshold).astype(int)
    for name, model in model_state["models"].items():
        score = model.predict(model_state["X_test"])
        fpr, tpr, _ = roc_curve(y_test_binary, score)
        auc_score = auc(fpr, tpr)
        roc_rows.append({"Model": name, "Group": model_state["model_groups"][name], "AUC": auc_score})
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} ({auc_score:.3f})"))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line={"dash": "dash"}))
    roc_fig.update_layout(title="ROC/AUC: 고부하 위험지역 분류 성능")
    st.plotly_chart(roc_fig, use_container_width=True)
    st.dataframe(pd.DataFrame(roc_rows).sort_values("AUC", ascending=False), use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("TableOne, t-SNE/UMAP, CCA, 상관분석")
    st.caption("평가 기준의 통계/부분집단 분석 항목을 한 화면에서 확인합니다.")
    table_cols = [
        "전기차_전체대수",
        "총_전력판매량",
        "총_판매수입",
        "전체_충전기대수",
        "총용량_kW",
        "인프라_부하지수",
        "전력_부하지수",
    ]
    table_kind, table_output = make_tableone(final_data, table_cols)
    if table_kind == "tableone":
        st.markdown(table_output)
    else:
        st.info("tableone 패키지가 없어 동일 변수에 대해 평균±표준편차와 t-test p-value를 직접 계산했습니다.")
        st.dataframe(table_output, use_container_width=True, hide_index=True)

    X_embed = model_state["X"].copy()
    y_embed = model_state["y"]
    scaled_embed = StandardScaler().fit_transform(X_embed)
    perplexity = max(5, min(20, len(X_embed) // 4))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", learning_rate="auto")
    tsne_xy = tsne.fit_transform(scaled_embed)
    embed_df = final_data[["지역", "용도"]].copy()
    embed_df["tSNE-1"] = tsne_xy[:, 0]
    embed_df["tSNE-2"] = tsne_xy[:, 1]
    embed_df["고부하"] = np.where(y_embed.values >= y_embed.quantile(0.7), "고부하", "일반")
    fig = px.scatter(embed_df, x="tSNE-1", y="tSNE-2", color="고부하", symbol="용도", hover_name="지역", title="t-SNE")
    st.plotly_chart(fig, use_container_width=True)

    try:
        import umap

        reducer = umap.UMAP(random_state=42)
        umap_xy = reducer.fit_transform(scaled_embed)
        title = "UMAP"
    except Exception:
        reducer = PCA(n_components=2, random_state=42)
        umap_xy = reducer.fit_transform(scaled_embed)
        title = "UMAP 패키지 미설치 시 PCA 대체 시각화"
    embed_df["UMAP-1"] = umap_xy[:, 0]
    embed_df["UMAP-2"] = umap_xy[:, 1]
    fig = px.scatter(embed_df, x="UMAP-1", y="UMAP-2", color="용도", symbol="고부하", hover_name="지역", title=title)
    st.plotly_chart(fig, use_container_width=True)

    corr_rows = []
    for x_col in ["전기차_전체대수", "전체_충전기대수", "인프라_부하지수"]:
        pearson_r, pearson_p = stats.pearsonr(final_data[x_col], final_data["전력_부하지수"])
        spearman_r, spearman_p = stats.spearmanr(final_data[x_col], final_data["전력_부하지수"])
        kendall_r, kendall_p = stats.kendalltau(final_data[x_col], final_data["전력_부하지수"])
        corr_rows.append(
            {
                "X": x_col,
                "Y": "전력_부하지수",
                "Pearson r": pearson_r,
                "Pearson p": pearson_p,
                "Spearman r": spearman_r,
                "Spearman p": spearman_p,
                "Kendall tau": kendall_r,
                "Kendall p": kendall_p,
            }
        )
    st.dataframe(pd.DataFrame(corr_rows).round(4), use_container_width=True, hide_index=True)

    cca_x = StandardScaler().fit_transform(final_data[["전기차_전체대수", "전체_충전기대수", "인프라_부하지수"]])
    cca_y = StandardScaler().fit_transform(final_data[["전력_부하지수"]])
    cca = CCA(n_components=1)
    x_c, y_c = cca.fit_transform(cca_x, cca_y)
    cca_df = pd.DataFrame({"CCA_X": x_c[:, 0], "CCA_Y": y_c[:, 0], "용도": final_data["용도"], "지역": final_data["지역"]})
    fig = px.scatter(cca_df, x="CCA_X", y="CCA_Y", color="용도", hover_name="지역", title="CCA canonical score")
    st.plotly_chart(fig, use_container_width=True)

with tabs[5]:
    st.subheader("SHAP summary / dependence / force plot + LIME 형태 설명")
    st.caption("SHAP 패키지가 있으면 실제 SHAP을 사용하고, 없으면 permutation/local contribution으로 대체합니다.")
    importance = model_state["importance"].head(12)
    fig = px.bar(importance.sort_values("Importance"), x="Importance", y="Feature", orientation="h", title="Permutation importance")
    st.plotly_chart(fig, use_container_width=True)

    selected_feature = st.selectbox("Dependence plot 변수", model_state["feature_columns"])
    local_region = st.selectbox("Force/LIME 형태로 볼 지역", sorted(final_data["지역"].unique()))
    local_usage = st.radio("Force/LIME 용도", ["자가용", "사업자용"], horizontal=True, key="force_usage")
    local_row = final_data[(final_data["지역"] == local_region) & (final_data["용도"] == local_usage)].head(1)
    local_x = make_feature_matrix(local_row).reindex(columns=model_state["feature_columns"], fill_value=0) if len(local_row) else None

    shap_ok = render_shap_or_fallback(model_state, selected_feature, local_x)

    best_model = model_state["models"][model_state["best_name"]]
    X_all = model_state["X"].copy()
    grid = np.linspace(X_all[selected_feature].quantile(0.05), X_all[selected_feature].quantile(0.95), 30)
    pd_rows = []
    for value in grid:
        X_tmp = X_all.copy()
        X_tmp[selected_feature] = value
        pd_rows.append({"Feature value": value, "Mean prediction": best_model.predict(X_tmp).mean()})
    fig = px.line(pd.DataFrame(pd_rows), x="Feature value", y="Mean prediction", markers=True, title="Partial dependence")
    st.plotly_chart(fig, use_container_width=True)

    if local_x is not None and len(local_x) > 0:
        baseline = best_model.predict(X_all).mean()
        pred_value = best_model.predict(local_x)[0]
        imp_map = model_state["importance"].set_index("Feature")["Importance"].reindex(model_state["feature_columns"]).fillna(0)
        centered = local_x.iloc[0] - X_all.mean()
        raw_contrib = centered * imp_map
        scale = (pred_value - baseline) / raw_contrib.sum() if raw_contrib.sum() != 0 else 0
        contrib = (raw_contrib * scale).sort_values(key=np.abs, ascending=False).head(8)
        force_df = pd.DataFrame({"Feature": contrib.index, "Contribution": contrib.values})
        fig = px.bar(force_df, x="Contribution", y="Feature", orientation="h", title="LIME/force 형태 지역별 기여도")
        fig.add_vline(x=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"기준 예측값 {baseline:,.2f}에서 선택 지역 예측값 {pred_value:,.2f}로 이동하는 방향을 보여줍니다.")

with tabs[6]:
    st.subheader("평가 조건 충족표")
    installed = {}
    for package, module_name in [("tableone", "tableone"), ("shap", "shap"), ("umap-learn", "umap")]:
        try:
            __import__(module_name)
            installed[package] = "설치됨"
        except Exception:
            installed[package] = "미설치: requirements.txt로 설치 권장"
    checklist = pd.DataFrame(
        [
            ["새로운 주제", "충족", "수도권 전기차 충전소 부하 예측 및 설치 시뮬레이션"],
            ["데이터 수집", "충족", "공공 CSV/Excel 8개 파일 사용, 환경부 월별 파일 추가"],
            ["TableOne", "조건부 충족", installed["tableone"]],
            ["t-SNE", "충족", "통계/군집 분석 탭"],
            ["UMAP", "조건부 충족", installed["umap-learn"]],
            ["CCA", "충족", "전기차/충전기/인프라 지수와 전력 부하지수"],
            ["correlation analysis 3개", "충족", "Pearson, Spearman, Kendall"],
            ["통계 분석", "충족", "자가용/사업자용 t-test, TableOne 형태"],
            ["머신러닝 5개", "충족", "RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, KNN"],
            ["딥러닝 2개", "충족", "Numpy_1D_CNN, Tabular_Transformer"],
            ["SHAP summary/dependence/force", "조건부 충족", installed["shap"]],
            ["LIME 형태 설명", "충족", "local contribution plot"],
            ["subgroup analysis", "충족", "자가용 vs 사업자용"],
            ["Streamlit web app", "충족", "현재 앱"],
            ["Residual/QQ/Actual-Pred", "충족", "예측 모델 비교 탭"],
        ],
        columns=["조건", "상태", "확인 위치/비고"],
    )
    st.dataframe(checklist, use_container_width=True, hide_index=True)
    st.markdown("#### 조건별 확인 요약")
    for _, row in checklist.iterrows():
        st.markdown(f"- **{row['조건']}**: {row['상태']} - {row['확인 위치/비고']}")
    st.markdown(
        """
        제출 전 더 안전하게 만들려면 새 컴퓨터에서 아래 명령을 한 번 실행하세요.

        ```powershell
        pip install -r requirements.txt
        ```

        그러면 TableOne, UMAP, SHAP 항목이 대체 구현이 아니라 실제 패키지 기반으로 표시됩니다.
        """
    )
