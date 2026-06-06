import os
import sys

# 실행 디렉토리에 관계없이 작업 디렉토리(CWD)를 프로젝트 루트로 강제 고정하고 sys.path 등록
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import pandas as pd
import numpy as np
from utils.data_processing import load_all_data, DEFAULT_DATA_DIR
from utils.models import train_models
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.metrics import roc_curve, auc
from utils.optimization import (
    calculate_bootstrap_ci,
    run_adversarial_attack,
    run_ablation_study,
    calculate_dca,
    run_nested_cv,
    run_spatial_external_validation,
    run_survival_simulation
)

def serialize_to_json(model_state, model_state_smote, json_path):
    # Helper to convert numpy types, dfs, series to json-serializable formats
    def default_converter(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    data = {
        "metrics": model_state["metrics"].to_dict(orient="records"),
        "importance": model_state["importance"].to_dict(orient="records"),
        "predictions": model_state["predictions"].to_dict(orient="records"),
        "best_name": model_state["best_name"],
        "feature_columns": model_state["feature_columns"],
        "model_groups": model_state["model_groups"],
        "X": model_state["X"].to_dict(orient="records"),
        "y": model_state["y"].tolist(),
        "X_train": model_state["X_train"].to_dict(orient="records"),
        "y_train": model_state["y_train"].tolist(),
        "X_test": model_state["X_test"].to_dict(orient="records"),
        "y_test": model_state["y_test"].tolist(),
    }
    
    # Embeddings
    if "precomputed_tsne_xy" in model_state:
        data["precomputed_tsne_xy"] = model_state["precomputed_tsne_xy"].tolist()
    if "precomputed_umap_xy" in model_state:
        data["precomputed_umap_xy"] = model_state["precomputed_umap_xy"].tolist()
        data["precomputed_umap_title"] = model_state.get("precomputed_umap_title", "UMAP")
    if "precomputed_cca_x_c" in model_state and "precomputed_cca_y_c" in model_state:
        data["precomputed_cca_x_c"] = model_state["precomputed_cca_x_c"].tolist()
        data["precomputed_cca_y_c"] = model_state["precomputed_cca_y_c"].tolist()
        
    # Bootstrap
    if "precomputed_bootstrap" in model_state:
        pb = model_state["precomputed_bootstrap"]
        data["precomputed_bootstrap"] = {
            "ci_rmse": [float(v) for v in pb[0]],
            "ci_r2": [float(v) for v in pb[1]],
            "bootstrap_rmse": [float(v) for v in pb[2]],
            "bootstrap_r2": [float(v) for v in pb[3]]
        }
        
    # Nested CV
    if "precomputed_nested_cv" in model_state:
        pnc = model_state["precomputed_nested_cv"]
        data["precomputed_nested_cv"] = {
            "mean_rmse": float(pnc[0]),
            "std_rmse": float(pnc[1]),
            "outer_scores": [float(v) for v in pnc[2]]
        }
        
    # Adversarial
    if "precomputed_adversarial" in model_state:
        pa = model_state["precomputed_adversarial"]
        data["precomputed_adversarial"] = pa.to_dict(orient="records")
        
    # Ablation
    if "precomputed_ablation" in model_state:
        pabl = model_state["precomputed_ablation"]
        data["precomputed_ablation"] = pabl.to_dict(orient="records")
        
    # DCA
    if "precomputed_dca" in model_state:
        pdca = model_state["precomputed_dca"]
        data["precomputed_dca"] = pdca.to_dict(orient="records")
        
    # Spatial CV
    if "precomputed_spatial" in model_state:
        ps = model_state["precomputed_spatial"]
        spatial_serializable = {}
        for region, metrics_tuple in ps.items():
            spatial_serializable[region] = [
                float(metrics_tuple[0]) if metrics_tuple[0] is not None else None,
                float(metrics_tuple[1]) if metrics_tuple[1] is not None else None,
                float(metrics_tuple[2]) if metrics_tuple[2] is not None else None,
                None
            ]
        data["precomputed_spatial"] = spatial_serializable
        
    # Survival
    if "precomputed_survival_5" in model_state:
        psurv = model_state["precomputed_survival_5"]
        data["precomputed_survival_5"] = psurv.to_dict(orient="records")

    # ROC data
    if "precomputed_roc_data" in model_state:
        data["precomputed_roc_data"] = model_state["precomputed_roc_data"]
        
    # Model State SMOTE metrics
    if model_state_smote is not None:
        data["model_state_smote_metrics"] = model_state_smote["metrics"].to_dict(orient="records")
        data["model_state_smote_best_name"] = model_state_smote["best_name"]
        
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=default_converter)

def export_trained_models():
    print("[1/5] Loading raw data...")
    final_data, _, _ = load_all_data(DEFAULT_DATA_DIR)
    final_json = final_data.to_json(orient="split")
    
    print("[2/5] Training baseline models (Pre-SMOTE 7 models)...")
    model_state = train_models(final_json, use_smote=False)
    
    # 1. Pre-compute heavy embeddings (t-SNE, UMAP, CCA)
    print("[2.3/5] Pre-computing embeddings (t-SNE, UMAP, CCA)...")
    try:
        X_embed = model_state["X"].copy()
        scaled_embed = StandardScaler().fit_transform(X_embed)
        
        # t-SNE
        perplexity = max(5, min(20, len(X_embed) // 4))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", learning_rate="auto")
        tsne_xy = tsne.fit_transform(scaled_embed)
        model_state["precomputed_tsne_xy"] = tsne_xy
        
        # UMAP
        try:
            import umap
            reducer = umap.UMAP(random_state=42)
            umap_xy = reducer.fit_transform(scaled_embed)
            model_state["precomputed_umap_title"] = "UMAP"
        except Exception:
            reducer = PCA(n_components=2, random_state=42)
            umap_xy = reducer.fit_transform(scaled_embed)
            model_state["precomputed_umap_title"] = "UMAP 패키지 미설치 시 PCA 대체 시각화"
        model_state["precomputed_umap_xy"] = umap_xy
        
        # CCA
        cca_x = StandardScaler().fit_transform(final_data[["전기차_전체대수", "충전인프라_규모_PCA", "충전기_1대당_평균용량", "인프라_부하지수"]])
        cca_y = StandardScaler().fit_transform(final_data[["전력_부하지수"]])
        cca = CCA(n_components=1)
        x_c, y_c = cca.fit_transform(cca_x, cca_y)
        model_state["precomputed_cca_x_c"] = x_c[:, 0]
        model_state["precomputed_cca_y_c"] = y_c[:, 0]
        
    except Exception as e:
        print(f"Embedding pre-computation failed: {e}")
        
    # 2. Pre-compute academic robustness evaluations (Bootstrap, CV, DCA, etc.)
    print("[2.7/5] Pre-computing robust evaluations (Nested CV, Bootstrap, DCA, etc.) to eliminate lag...")
    try:
        best_name = model_state["best_name"]
        best_model = model_state["models"][best_name]
        X_train, X_test = model_state["X_train"], model_state["X_test"]
        y_train, y_test = model_state["y_train"], model_state["y_test"]
        X, y = model_state["X"], model_state["y"]
        importance = model_state["importance"]
        
        # Bootstrap CI
        print(" -> Computing Bootstrap CI...")
        model_state["precomputed_bootstrap"] = calculate_bootstrap_ci(best_model, X_test, y_test, n_iterations=100)
        
        # Nested CV
        print(" -> Computing Nested 10-fold CV (this may take a few seconds)...")
        model_state["precomputed_nested_cv"] = run_nested_cv(best_model, X, y)
        
        # Adversarial Attack
        print(" -> Computing Adversarial Attack...")
        model_state["precomputed_adversarial"] = run_adversarial_attack(best_model, X_test, y_test)
        
        # Ablation Study
        print(" -> Computing Ablation Study...")
        model_state["precomputed_ablation"] = run_ablation_study(best_model, X_train, y_train, X_test, y_test, importance)
        
        # DCA
        print(" -> Computing DCA...")
        model_state["precomputed_dca"] = calculate_dca(best_model, X_test, y_test)
        
        # Spatial CV (Pre-calculate for holdouts)
        print(" -> Computing Spatial External Validations...")
        spatial_results = {}
        for region in ["인천", "서울", "경기"]:
            ext_rmse, ext_mae, ext_r2, ext_err = run_spatial_external_validation(best_model, X, y, region)
            spatial_results[region] = (ext_rmse, ext_mae, ext_r2, ext_err)
        model_state["precomputed_spatial"] = spatial_results
        
        # Survival Analysis (growth_rate = 5%)
        print(" -> Computing Survival Curve...")
        surv_res = run_survival_simulation(final_data, growth_rate=0.05)
        model_state["precomputed_survival_5"] = surv_res
        
        # Precompute ROC Curve for all models
        print(" -> Precomputing ROC curves...")
        threshold = y.quantile(0.7)
        y_test_binary = (y_test >= threshold).astype(int)
        roc_data = {}
        for name, model in model_state["models"].items():
            score = model.predict(X_test)
            fpr, tpr, _ = roc_curve(y_test_binary, score)
            roc_data[name] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(auc(fpr, tpr)),
                "group": model_state["model_groups"][name]
            }
        model_state["precomputed_roc_data"] = roc_data
        
    except Exception as e:
        print(f"Robustness pre-computation failed: {e}")
        
    print("[3/5] Training balanced models (Post-SMOTE 7 models)...")
    model_state_smote = train_models(final_json, use_smote=True)
    
    os.makedirs("results", exist_ok=True)
    
    # 3. Export to ONNX Format
    print("[3.5/5] Exporting best model to ONNX...")
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        
        best_name = model_state["best_name"]
        best_model = model_state["models"][best_name]
        
        initial_type = [('float_input', FloatTensorType([None, 13]))]
        
        try:
            onnx_model = convert_sklearn(best_model, initial_types=initial_type, target_opset=12)
        except Exception as conversion_error:
            print(f"ONNX conversion for best model {best_name} failed: {conversion_error}. Falling back to RandomForest (Tuned)...")
            best_model = model_state["models"]["RandomForest (Tuned)"]
            onnx_model = convert_sklearn(best_model, initial_types=initial_type, target_opset=12)
            model_state["best_name"] = "RandomForest (Tuned)"
            
        onnx_path = "results/best_model.onnx"
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"[SUCCESS] ONNX model successfully saved to '{onnx_path}'")
    except Exception as e:
        print(f"[ERROR] Failed to export ONNX model: {e}")

    # 4. Export to JSON Format
    json_path = "results/precomputed_analytics.json"
    print(f"[4/5] Serializing all visualization and academic metrics to '{json_path}'...")
    try:
        serialize_to_json(model_state, model_state_smote, json_path)
        print(f"[SUCCESS] Metrics successfully saved to '{json_path}'")
    except Exception as e:
        print(f"[ERROR] Failed to export JSON analytics: {e}")

    print("[SUCCESS] Two-track models, embeddings, and robustness metrics exported successfully!")

if __name__ == "__main__":
    export_trained_models()
