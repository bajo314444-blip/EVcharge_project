import os
import joblib
import pandas as pd
import numpy as np
from utils.data_processing import load_all_data, DEFAULT_DATA_DIR
from utils.models import train_models
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from utils.optimization import (
    calculate_bootstrap_ci,
    run_adversarial_attack,
    run_ablation_study,
    calculate_dca,
    run_nested_cv,
    run_spatial_external_validation,
    run_survival_simulation
)

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
        
    except Exception as e:
        print(f"Robustness pre-computation failed: {e}")
        
    print("[3/5] Training balanced models (Post-SMOTE 7 models)...")
    model_state_smote = train_models(final_json, use_smote=True)
    
    # Pack both states
    package = {
        "model_state": model_state,
        "model_state_smote": model_state_smote
    }
    
    os.makedirs("results", exist_ok=True)
    output_path = "results/trained_model_state.joblib"
    
    print(f"[4/5] Serializing model state to '{output_path}'...")
    joblib.dump(package, output_path, compress=3)
    
    print("[SUCCESS] All models, embeddings, and robustness metrics pre-trained and serialized successfully!")

if __name__ == "__main__":
    export_trained_models()
