import os
import sys

# 실행 디렉토리에 관계없이 작업 디렉토리(CWD)를 프로젝트 루트로 강제 고정하고 sys.path 등록
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy import stats
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from utils.data_processing import load_all_data, DEFAULT_DATA_DIR
from utils.models import make_feature_matrix, train_models

# Force default colored template
pio.templates.default = "plotly"

def save_baseline_figures():
    print("Loading data...")
    final_data, monthly_data, _ = load_all_data(DEFAULT_DATA_DIR)
    
    print("Training models to generate figures...")
    model_state = train_models(final_data.to_json(orient="split"))
    
    fig_dir = "results/baseline/figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    # Define a clear color map for usage to ensure colors are applied
    cmap = {"자가용": "#00A699", "사업자용": "#FF5A5F"}
    
    print("Generating Test RMSE comparison...")
    metrics = model_state["metrics"]
    test_metrics = metrics[metrics["Split"] == "Test"].sort_values("RMSE")
    fig = px.bar(
        test_metrics, 
        x="Model", 
        y="RMSE", 
        color="Group", 
        title="Test RMSE 비교", 
        text_auto=".1f",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.write_image(f"{fig_dir}/01_test_rmse.png", scale=2)

    print("Generating Actual vs Predicted...")
    pred = model_state["predictions"]
    best_pred = pred[pred["Model"] == model_state["best_name"]].copy()
    fig = px.scatter(
        best_pred, 
        x="Actual", 
        y="Predicted", 
        color="용도", 
        hover_name="지역", 
        title="Actual vs Prediction",
        color_discrete_map=cmap
    )
    min_v = min(best_pred["Actual"].min(), best_pred["Predicted"].min())
    max_v = max(best_pred["Actual"].max(), best_pred["Predicted"].max())
    fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", name="완전 일치", line=dict(color='orange')))
    fig.write_image(f"{fig_dir}/02_actual_vs_predicted.png", scale=2)
    
    print("Generating Residual Plot...")
    best_pred["Residual"] = best_pred["Actual"] - best_pred["Predicted"]
    fig = px.scatter(
        best_pred, 
        x="Predicted", 
        y="Residual", 
        color="용도", 
        hover_name="지역", 
        title="Residual Plot",
        color_discrete_map=cmap
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.write_image(f"{fig_dir}/03_residual_plot.png", scale=2)
    
    print("Generating QQ Plot...")
    qq = stats.probplot(best_pred["Residual"], dist="norm")
    qq_df = pd.DataFrame({"Theoretical": qq[0][0], "Ordered residual": qq[0][1]})
    fig = px.scatter(qq_df, x="Theoretical", y="Ordered residual", title="QQ Plot")
    fig.update_traces(marker=dict(color="#636EFA"))
    fig.write_image(f"{fig_dir}/04_qq_plot.png", scale=2)
    
    print("Generating ROC/AUC...")
    threshold = model_state["y"].quantile(0.7)
    roc_fig = go.Figure()
    y_test_binary = (model_state["y_test"] >= threshold).astype(int)
    colors = px.colors.qualitative.D3
    for i, (name, model) in enumerate(model_state["models"].items()):
        score = model.predict(model_state["X_test"])
        fpr, tpr, _ = roc_curve(y_test_binary, score)
        auc_score = auc(fpr, tpr)
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} ({auc_score:.3f})", line=dict(color=colors[i % len(colors)])))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="black")))
    roc_fig.update_layout(title="ROC/AUC: 고부하 위험지역 분류 성능")
    roc_fig.write_image(f"{fig_dir}/05_roc_auc.png", scale=2)
    
    print("Generating t-SNE...")
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
    fig = px.scatter(
        embed_df, 
        x="tSNE-1", 
        y="tSNE-2", 
        color="고부하", 
        symbol="용도", 
        hover_name="지역", 
        title="t-SNE",
        color_discrete_map={"고부하": "#FF5A5F", "일반": "#00A699"}
    )
    fig.write_image(f"{fig_dir}/06_tsne.png", scale=2)
    
    print("Generating CCA...")
    cca_x = StandardScaler().fit_transform(final_data[["전기차_전체대수", "전체_충전기대수", "인프라_부하지수"]])
    cca_y = StandardScaler().fit_transform(final_data[["전력_부하지수"]])
    cca = CCA(n_components=1)
    x_c, y_c = cca.fit_transform(cca_x, cca_y)
    cca_df = pd.DataFrame({"CCA_X": x_c[:, 0], "CCA_Y": y_c[:, 0], "용도": final_data["용도"], "지역": final_data["지역"]})
    fig = px.scatter(
        cca_df, 
        x="CCA_X", 
        y="CCA_Y", 
        color="용도", 
        hover_name="지역", 
        title="CCA canonical score",
        color_discrete_map=cmap
    )
    fig.write_image(f"{fig_dir}/07_cca.png", scale=2)
    
    print("Generating Permutation Importance...")
    importance = model_state["importance"].head(12)
    fig = px.bar(
        importance.sort_values("Importance"), 
        x="Importance", 
        y="Feature", 
        orientation="h", 
        title="Permutation importance",
        color="Importance",
        color_continuous_scale="Viridis"
    )
    fig.write_image(f"{fig_dir}/08_importance.png", scale=2)
    
    print("Generating Partial Dependence...")
    selected_feature = model_state["feature_columns"][0]
    best_model = model_state["models"][model_state["best_name"]]
    X_all = model_state["X"].copy()
    grid = np.linspace(X_all[selected_feature].quantile(0.05), X_all[selected_feature].quantile(0.95), 30)
    pd_rows = []
    for value in grid:
        X_tmp = X_all.copy()
        X_tmp[selected_feature] = value
        pd_rows.append({"Feature value": value, "Mean prediction": best_model.predict(X_tmp).mean()})
    fig = px.line(pd.DataFrame(pd_rows), x="Feature value", y="Mean prediction", markers=True, title=f"Partial dependence: {selected_feature}")
    fig.update_traces(line=dict(color="#EF553B"), marker=dict(color="#EF553B"))
    fig.write_image(f"{fig_dir}/09_partial_dependence.png", scale=2)
    
    print("All colored figures saved successfully to results/baseline/figures/")

if __name__ == "__main__":
    save_baseline_figures()
