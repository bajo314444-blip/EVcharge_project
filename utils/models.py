import numpy as np
import pandas as pd
import streamlit as st
from io import StringIO
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import warnings

def smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class NumpyCNN1DRegressor:
    """Small 1D-CNN regressor for tabular feature vectors."""

    def __init__(self, n_filters=10, kernel_size=3, epochs=700, lr=0.015, random_state=42):
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0) + 1e-8
        self.y_mean_ = y.mean()
        self.y_std_ = y.std() + 1e-8
        Xs = (X - self.x_mean_) / self.x_std_
        ys = (y - self.y_mean_) / self.y_std_

        rng = np.random.default_rng(self.random_state)
        n_features = Xs.shape[1]
        self.kernel_size_ = min(self.kernel_size, n_features)
        n_pos = n_features - self.kernel_size_ + 1
        self.w_conv_ = rng.normal(0, 0.12, size=(self.n_filters, self.kernel_size_))
        self.b_conv_ = np.zeros(self.n_filters)
        self.w_out_ = rng.normal(0, 0.12, size=n_pos * self.n_filters)
        self.b_out_ = 0.0

        for _ in range(self.epochs):
            z = self._conv_forward(Xs)
            h = np.maximum(z, 0)
            flat = h.reshape(len(Xs), -1)
            pred = flat @ self.w_out_ + self.b_out_
            grad_pred = 2 * (pred - ys) / len(Xs)

            grad_w_out = flat.T @ grad_pred
            grad_b_out = grad_pred.sum()
            grad_flat = np.outer(grad_pred, self.w_out_)
            grad_h = grad_flat.reshape(h.shape)
            grad_z = grad_h * (z > 0)

            grad_w_conv = np.zeros_like(self.w_conv_)
            grad_b_conv = grad_z.sum(axis=(0, 1))
            for pos in range(n_pos):
                window = Xs[:, pos : pos + self.kernel_size_]
                grad_w_conv += grad_z[:, pos, :].T @ window

            self.w_out_ -= self.lr * grad_w_out
            self.b_out_ -= self.lr * grad_b_out
            self.w_conv_ -= self.lr * grad_w_conv
            self.b_conv_ -= self.lr * grad_b_conv
        return self

    def _conv_forward(self, Xs):
        n_pos = Xs.shape[1] - self.kernel_size_ + 1
        out = np.zeros((len(Xs), n_pos, self.n_filters))
        for pos in range(n_pos):
            window = Xs[:, pos : pos + self.kernel_size_]
            out[:, pos, :] = window @ self.w_conv_.T + self.b_conv_
        return out

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xs = (X - self.x_mean_) / self.x_std_
        h = np.maximum(self._conv_forward(Xs), 0).reshape(len(Xs), -1)
        pred = h @ self.w_out_ + self.b_out_
        return pred * self.y_std_ + self.y_mean_

    def score(self, X, y):
        return r2_score(y, self.predict(X))

class TabularTransformerRegressor:
    """Lightweight transformer-style feature encoder plus ridge regression head."""

    def __init__(self, d_model=8, random_state=42, ridge_alpha=1.0):
        self.d_model = d_model
        self.random_state = random_state
        self.ridge_alpha = ridge_alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0) + 1e-8
        rng = np.random.default_rng(self.random_state)
        self.n_features_ = X.shape[1]
        self.token_weight_ = rng.normal(0, 0.35, size=(self.n_features_, self.d_model))
        self.feature_embed_ = rng.normal(0, 0.15, size=(self.n_features_, self.d_model))
        self.w_q_ = rng.normal(0, 0.25, size=(self.d_model, self.d_model))
        self.w_k_ = rng.normal(0, 0.25, size=(self.d_model, self.d_model))
        self.w_v_ = rng.normal(0, 0.25, size=(self.d_model, self.d_model))
        z = self._encode((X - self.x_mean_) / self.x_std_)
        self.head_ = Ridge(alpha=self.ridge_alpha)
        self.head_.fit(z, y)
        return self

    def _encode(self, Xs):
        tokens = Xs[:, :, None] * self.token_weight_[None, :, :] + self.feature_embed_[None, :, :]
        q = tokens @ self.w_q_
        k = tokens @ self.w_k_
        v = tokens @ self.w_v_
        attn = softmax((q @ np.swapaxes(k, 1, 2)) / np.sqrt(self.d_model), axis=-1)
        encoded = attn @ v
        pooled = encoded.mean(axis=1)
        flat = encoded.reshape(len(Xs), -1)
        return np.hstack([Xs, pooled, flat])

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        z = self._encode((X - self.x_mean_) / self.x_std_)
        return self.head_.predict(z)

    def score(self, X, y):
        return r2_score(y, self.predict(X))

def make_feature_matrix(final):
    ml = final.copy()
    ml = pd.get_dummies(ml, columns=["시도", "용도"], dtype=float)
    feature_cols = ["전기차_전체대수", "충전인프라_규모_PCA", "충전기_1대당_평균용량", "인프라_부하지수"]
    feature_cols += [c for c in ml.columns if c.startswith("시도_") or c.startswith("용도_")]
    X = ml[feature_cols].rename(
        columns={
            "전기차_전체대수": "total_ev_count",
            "충전인프라_규모_PCA": "infra_size_pca",
            "충전기_1대당_평균용량": "avg_capacity_per_charger",
            "인프라_부하지수": "infra_load_index",
            "시도_경기": "region_gyeonggi",
            "시도_서울": "region_seoul",
            "시도_인천": "region_incheon",
            "용도_사업자용": "usage_business",
            "용도_자가용": "usage_private",
        }
    )
    
    # 기본 Time Spike 컬럼 추가 (0으로 초기화)
    for col in ["is_commute_time", "is_holiday", "is_golden_week", "is_weekend"]:
        if col not in X.columns:
            X[col] = 0.0
            
    return X

def apply_smote_for_regression(X_train, y_train, n_bins=5):
    bins = np.quantile(y_train, np.linspace(0, 1, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf
    y_binned = np.digitize(y_train, bins)
    
    class_counts = pd.Series(y_binned).value_counts()
    min_count = class_counts.min()
    if min_count < 2:
        return X_train, y_train
        
    k_neighbors = min(5, min_count - 1)
    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
    
    try:
        X_resampled, _ = smote.fit_resample(X_train, y_binned)
        knn = KNeighborsRegressor(n_neighbors=k_neighbors)
        knn.fit(X_train, y_train)
        y_resampled = knn.predict(X_resampled)
        if isinstance(y_train, pd.Series):
            y_resampled = pd.Series(y_resampled)
        y_resampled[:len(y_train)] = y_train
        return X_resampled, y_resampled
    except ValueError:
        return X_train, y_train

@st.cache_data(show_spinner="예측 모델을 학습하고 평가하는 중입니다...")
def train_models(final_json, use_smote=False):
    final = pd.read_json(StringIO(final_json), orient="split")
    X = make_feature_matrix(final)
    y = final["전력_부하지수"].astype(float)
    row_info = final[["지역", "용도", "전력_부하지수"]].copy()

    X_train, X_temp, y_train, y_temp, info_train, info_temp = train_test_split(
        X, y, row_info, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test, info_val, info_test = train_test_split(
        X_temp, y_temp, info_temp, test_size=1 / 3, random_state=42
    )

    if use_smote:
        X_train, y_train = apply_smote_for_regression(X_train, y_train)
        
    # Time Spike Data Augmentation (가상 시나리오 가중치 학습)
    X_commute = X_train.copy()
    X_commute["is_commute_time"] = 1.0
    y_commute = y_train * 1.3
    
    X_weekend = X_train.copy()
    X_weekend["is_weekend"] = 1.0
    y_weekend = y_train * 1.5
    
    X_holiday = X_train.copy()
    X_holiday["is_holiday"] = 1.0
    X_holiday["is_golden_week"] = 1.0
    y_holiday = y_train * 2.0
    
    X_train = pd.concat([X_train, X_commute, X_weekend, X_holiday], ignore_index=True)
    y_train = pd.concat([y_train, y_commute, y_weekend, y_holiday], ignore_index=True)
    
    # 셔플링
    idx = np.random.permutation(len(X_train))
    X_train = X_train.iloc[idx].reset_index(drop=True)
    y_train = y_train.iloc[idx].reset_index(drop=True)

    rf_base = RandomForestRegressor(random_state=42)
    rf_param_grid = {'max_depth': [3, 5, 10], 'min_samples_leaf': [1, 2, 4]}
    rf_grid = GridSearchCV(rf_base, rf_param_grid, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='neg_mean_squared_error')
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_

    gb_base = GradientBoostingRegressor(random_state=42)
    gb_param_grid = {'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]}
    gb_grid = GridSearchCV(gb_base, gb_param_grid, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='neg_mean_squared_error')
    gb_grid.fit(X_train, y_train)
    best_gb = gb_grid.best_estimator_

    models = {
        "RandomForest (Tuned)": ("Machine Learning", best_rf),
        "ExtraTrees": ("Machine Learning", ExtraTreesRegressor(max_depth=5, min_samples_leaf=2, random_state=42)),
        "GradientBoosting (Tuned)": ("Machine Learning", best_gb),
        "HistGradientBoosting": ("Machine Learning", HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=42)),
        "KNN": ("Machine Learning", Pipeline([("scale", MinMaxScaler()), ("model", KNeighborsRegressor(n_neighbors=5))])),
        "Numpy_1D_CNN": ("Deep Learning - CNN", NumpyCNN1DRegressor(random_state=42)),
        "Tabular_Transformer": ("Deep Learning - Transformer", TabularTransformerRegressor(random_state=42)),
    }

    rows = []
    predictions = []
    fitted = {}
    model_groups = {}
    for name, (group, model) in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        model_groups[name] = group
        for split, X_part, y_part in [
            ("Train", X_train, y_train),
            ("Validation", X_val, y_val),
            ("Test", X_test, y_test),
        ]:
            pred = model.predict(X_part)
            rows.append(
                {
                    "Model": name,
                    "Group": group,
                    "Split": split,
                    "RMSE": float(np.sqrt(mean_squared_error(y_part, pred))),
                    "MAE": float(mean_absolute_error(y_part, pred)),
                    "MSE": float(mean_squared_error(y_part, pred)),
                    "SMAPE(%)": float(smape(y_part, pred)),
                    "R2": float(r2_score(y_part, pred)) if len(y_part) > 1 else np.nan,
                }
            )
            if split == "Test":
                pred_df = info_test.copy()
                pred_df["Model"] = name
                pred_df["Group"] = group
                pred_df["Actual"] = y_part.values
                pred_df["Predicted"] = pred
                predictions.append(pred_df)

    metrics = pd.DataFrame(rows)
    pred_all = pd.concat(predictions, ignore_index=True)
    best_name = metrics[metrics["Split"] == "Test"].sort_values("RMSE").iloc[0]["Model"]
    best_model = fitted[best_name]

    perm = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=42,
        scoring="neg_mean_squared_error",
    )
    importance = (
        pd.DataFrame({"Feature": X.columns, "Importance": perm.importances_mean})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "metrics": metrics,
        "predictions": pred_all,
        "best_name": best_name,
        "model_groups": model_groups,
        "importance": importance,
        "feature_columns": list(X.columns),
        "X": X,
        "y": y,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "info_test": info_test,
        "models": fitted,
    }
