import numpy as np
import pandas as pd
from scipy.optimize import linprog

def optimize_highway_chargers(hw_df, budget):
    """
    고속도로 휴게소/IC 최적 입지 선정 알고리즘 (Linear Programming 기반)
    주어진 예산(budget) 내에서 부하 감소 효율성이 가장 높은 노드 조합을 찾습니다.
    
    LP Formulation:
    Maximize sum( Delta_L_i * X_i )  => Minimize sum( -Delta_L_i * X_i )
    Subject to:
        sum(X_i) <= budget
        0 <= X_i <= Max_Capacity_i
    """
    n_nodes = len(hw_df)
    if n_nodes == 0 or budget <= 0:
        hw_df = hw_df.copy()
        hw_df["최적_추가대수"] = 0
        hw_df["최적화후_부하점수"] = hw_df.get("부하_예측점수", 0)
        return hw_df
        
    # 부하_예측점수가 없을 경우(초기화 에러 방지) 임의 점수 사용
    if "부하_예측점수" not in hw_df.columns:
        hw_df["부하_예측점수"] = np.random.uniform(30, 90, n_nodes)
        
    # 각 노드별 충전기 1대 추가 시의 부하 감소량 (효율성)
    # 기존 점수의 약 2~8%가 감소한다고 가정 (노드 용량별 차등)
    delta_L = hw_df["부하_예측점수"].values * np.random.uniform(0.02, 0.08, n_nodes)
    
    # LP Solver 적용 (scipy.optimize.linprog)
    c = -delta_L  # linprog는 minimize를 수행하므로 음수로 변환
    
    A_ub = np.ones((1, n_nodes))
    b_ub = np.array([budget])
    
    # 각 노드별 최대 설치 가능 대수(Max_Capacity) 상한선(Upper Bound) 제약
    bounds = [(0, cap) for cap in hw_df["Max_Capacity"]]
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    hw_df = hw_df.copy()
    if res.success:
        # 결과값을 정수로 반올림
        hw_df["최적_추가대수"] = np.round(res.x).astype(int)
    else:
        hw_df["최적_추가대수"] = 0
        
    # 최적화 후의 부하 점수 갱신
    hw_df["최적화후_부하점수"] = np.clip(
        hw_df["부하_예측점수"] - (delta_L * hw_df["최적_추가대수"]), 
        0, 100
    )
    
    return hw_df


# ==========================================
# Robustness & Model Evaluation Algorithms
# ==========================================
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import copy

def calculate_bootstrap_ci(best_model, X_test, y_test, n_iterations=100):
    """
    Bootstrap 95% Confidence Interval
    """
    rmse_scores = []
    r2_scores = []
    
    n_size = len(y_test)
    X_test_arr = np.array(X_test)
    y_test_arr = np.array(y_test)
    
    for i in range(n_iterations):
        np.random.seed(42 + i)
        indices = np.random.choice(range(n_size), size=n_size, replace=True)
        X_sample = X_test_arr[indices]
        y_sample = y_test_arr[indices]
        
        preds = best_model.predict(X_sample)
        rmse = np.sqrt(mean_squared_error(y_sample, preds))
        r2 = r2_score(y_sample, preds)
        
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        
    ci_rmse = (np.percentile(rmse_scores, 2.5), np.percentile(rmse_scores, 97.5))
    ci_r2 = (np.percentile(r2_scores, 2.5), np.percentile(r2_scores, 97.5))
    
    return ci_rmse, ci_r2, rmse_scores, r2_scores


def run_adversarial_attack(best_model, X_test, y_test, noise_levels=[0.05, 0.1, 0.2]):
    """
    Adversarial Attack Analysis: Inject Gaussian Noise
    """
    results = []
    base_preds = best_model.predict(X_test)
    base_rmse = np.sqrt(mean_squared_error(y_test, base_preds))
    
    results.append({"Noise_Level": "0% (Base)", "RMSE": base_rmse, "Drop_Ratio(%)": 0.0})
    
    X_test_arr = np.array(X_test)
    stds = np.std(X_test_arr, axis=0)
    
    for noise in noise_levels:
        np.random.seed(42)
        noise_matrix = np.random.normal(0, stds * noise, size=X_test_arr.shape)
        X_adv = X_test_arr + noise_matrix
        
        preds = best_model.predict(X_adv)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        drop_ratio = (rmse - base_rmse) / base_rmse * 100
        
        results.append({"Noise_Level": f"{int(noise*100)}%", "RMSE": rmse, "Drop_Ratio(%)": drop_ratio})
        
    return pd.DataFrame(results)


def run_ablation_study(best_model, X_train, y_train, X_test, y_test, importances):
    """
    Feature importance ablation study (Sensitivity analysis)
    """
    from sklearn.base import clone
    features_ordered = importances["Feature"].tolist()
    
    results = []
    current_features = features_ordered.copy()
    
    for i in range(len(features_ordered) - 1):
        if len(current_features) < 2:
            break
            
        try:
            model_clone = clone(best_model)
        except:
            model_clone = copy.deepcopy(best_model)
            
        model_clone.fit(X_train[current_features], y_train)
        preds = model_clone.predict(X_test[current_features])
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        results.append({
            "Num_Features": len(current_features), 
            "RMSE": rmse, 
            "Removed_Feature": features_ordered[-i] if i > 0 else "None"
        })
        
        current_features.pop() # Remove least important
        
    return pd.DataFrame(results)


def calculate_dca(best_model, X_test, y_test):
    """
    Decision Curve Analysis adapted for Regression.
    """
    preds_continuous = best_model.predict(X_test)
    
    thresholds = np.linspace(np.percentile(y_test, 50), np.percentile(y_test, 90), 20)
    net_benefits = []
    
    for th in thresholds:
        y_true_bin = (y_test >= th).astype(int)
        pred_bin = (preds_continuous >= th).astype(int)
        
        tp = np.sum((y_true_bin == 1) & (pred_bin == 1))
        fp = np.sum((y_true_bin == 0) & (pred_bin == 1))
        n = len(y_test)
        
        prevalence = np.sum(y_true_bin) / n
        if prevalence == 0 or prevalence == 1:
            continue
            
        pt = prevalence 
        
        nb_model = (tp / n) - (fp / n) * (pt / (1 - pt))
        
        tp_all = np.sum(y_true_bin)
        fp_all = n - tp_all
        nb_all = (tp_all / n) - (fp_all / n) * (pt / (1 - pt))
        
        net_benefits.append({"Threshold_Value": th, "Model_NB": nb_model, "Treat_All_NB": nb_all, "Treat_None_NB": 0.0})
        
    return pd.DataFrame(net_benefits)


def run_survival_simulation(final_data, growth_rate=0.05):
    """
    Simulate Time-to-Overload based on EV growth rate.
    """
    df = final_data.copy()
    critical_threshold = df["전력_부하지수"].quantile(0.8) # Top 20%
    
    years = range(1, 16) # up to 15 years
    survival_data = []
    
    for i, row in df.iterrows():
        base_load = row["총_전력판매량"]
        capacity = row["총용량_kW"]
        
        overload_year = 15
        event = 0
        
        if capacity > 0:
            current_index = base_load / capacity
            if current_index >= critical_threshold:
                overload_year = 0
                event = 1
            else:
                for y in years:
                    simulated_load = base_load * ((1 + growth_rate) ** y)
                    simulated_index = simulated_load / capacity
                    if simulated_index >= critical_threshold:
                        overload_year = y
                        event = 1
                        break
                        
        survival_data.append({"Region": row["지역"], "Time_to_Overload": overload_year, "Event": event})
        
    return pd.DataFrame(survival_data)


def calculate_single_region_trajectory(base_load, capacity, growth_rate, added_kw, critical_threshold):
    """
    Calculate the Load Index trajectory over 15 years for a single region,
    before and after adding charging capacity.
    """
    trajectory = []
    
    overload_year_before = 15
    overload_year_after = 15
    
    for y in range(16):
        # 1. EV load grows exponentially
        simulated_load = base_load * ((1 + growth_rate) ** y)
        
        # 2. Base Index
        base_index = simulated_load / capacity if capacity > 0 else 0
        if base_index >= critical_threshold and overload_year_before == 15 and y > 0:
            if simulated_load / capacity >= critical_threshold: # Ensure strictly evaluated
                pass
            
        if y == 0 and base_index >= critical_threshold:
            overload_year_before = 0
        elif y > 0 and base_index >= critical_threshold and overload_year_before == 15:
            overload_year_before = y
            
        # 3. After Index
        after_capacity = capacity + added_kw
        after_index = simulated_load / after_capacity if after_capacity > 0 else 0
        
        if y == 0 and after_index >= critical_threshold:
            overload_year_after = 0
        elif y > 0 and after_index >= critical_threshold and overload_year_after == 15:
            overload_year_after = y
            
        trajectory.append({
            "Year": y,
            "상태": "설치 전",
            "부하지수": base_index
        })
        trajectory.append({
            "Year": y,
            "상태": "설치 후",
            "부하지수": after_index
        })
        
    return pd.DataFrame(trajectory), overload_year_before, overload_year_after


def run_nested_cv(best_model, X, y, n_splits=10):
    """
    Perform Nested 10-fold Cross-Validation for the best model to rigorously evaluate performance.
    Since best_model is already the best_estimator_, we deepcopy it.
    """
    import copy
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error
    import numpy as np
    
    # We use deepcopy because custom models like NumpyCNN1DRegressor cannot be cloned by sklearn
    base_estimator = copy.deepcopy(best_model)
    
    if isinstance(base_estimator, RandomForestRegressor):
        param_grid = {'max_depth': [3, 5, 10], 'min_samples_leaf': [1, 2]}
    elif isinstance(base_estimator, GradientBoostingRegressor):
        param_grid = {'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]}
    else:
        param_grid = {} # No hyperparameter tuning if we don't know the grid

    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    
    outer_scores = []
    
    X_arr = np.array(X)
    y_arr = np.array(y)
    
    for train_idx, test_idx in outer_cv.split(X_arr):
        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]
        
        if param_grid:
            grid = GridSearchCV(base_estimator, param_grid, cv=inner_cv, scoring='neg_mean_squared_error', n_jobs=-1)
            grid.fit(X_tr, y_tr)
            best_local_model = grid.best_estimator_
        else:
            best_local_model = copy.deepcopy(base_estimator)
            best_local_model.fit(X_tr, y_tr)
            
        preds = best_local_model.predict(X_te)
        rmse = np.sqrt(mean_squared_error(y_te, preds))
        outer_scores.append(rmse)
        
    return np.mean(outer_scores), np.std(outer_scores), outer_scores


def run_spatial_external_validation(best_model, X, y, holdout_region="인천"):
    """
    Perform Spatial External Validation by holding out a specific region completely from training.
    """
    import copy
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np

    # Create mask based on the selected region
    if holdout_region == "서울":
        test_mask = X["region_seoul"] == 1
    elif holdout_region == "경기":
        test_mask = X["region_gyeonggi"] == 1
    elif holdout_region == "인천":
        test_mask = X["region_incheon"] == 1
    else:
        raise ValueError(f"Unknown region: {holdout_region}")
        
    train_mask = ~test_mask
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    if len(X_test) == 0:
        return None, None, None, "선택한 배제 지역에 해당하는 데이터가 없습니다."
        
    if len(X_train) == 0:
        return None, None, None, "학습용 데이터가 부족합니다."

    model = copy.deepcopy(best_model)
    
    # We don't tune hyperparams here for speed, just fit with the best params found on internal validation
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds) if len(y_test) > 1 else np.nan
    
    return rmse, mae, r2, None


# ==========================================
# Premium Control Module Functions (V4.8)
# ==========================================

def simulate_dynamic_pricing(hourly_series, elasticity=-0.2, peak_surcharge=0.0, discount_rate=0.0, peak_hours=None, off_peak_hours=None):
    """
    가격 탄력성을 반영하여 시간대별 수요를 분산시키는 시뮬레이션 엔진.
    - hourly_series: pd.Series 또는 np.ndarray (24개 시간대의 충전 부하 데이터)
    - elasticity: 가격 탄력성 계수 (기본값 -0.2, 비탄력적)
    - peak_surcharge: peak 시간대 요금 할증률 (예: 0.20 = 20%)
    - discount_rate: off-peak 시간대 요금 할인율 (예: 0.15 = 15%)
    """
    if peak_hours is None:
        peak_hours = [10, 11, 13, 14, 15, 16, 18, 19, 20, 21]
    if off_peak_hours is None:
        off_peak_hours = [23, 0, 1, 2, 3, 4, 5, 6, 7, 8]
        
    hourly_array = np.array(hourly_series, dtype=float)
    n_hours = len(hourly_array)
    if n_hours != 24:
        raise ValueError("시간대별 부하 데이터는 24개 요소로 이루어져야 합니다.")
        
    original_total = np.sum(hourly_array)
    if original_total == 0:
        return hourly_array, np.zeros(24)
        
    # Calculate price change vector
    price_change = np.zeros(24)
    for h in range(24):
        if h in peak_hours:
            price_change[h] = peak_surcharge
        elif h in off_peak_hours:
            price_change[h] = -discount_rate
            
    # Calculate change in demand for each hour based on elasticity
    delta_demand = hourly_array * price_change * elasticity
    
    # Reductions in peak hours (where price_change > 0)
    reductions = np.minimum(delta_demand, 0.0)
    total_reduced_demand = -np.sum(reductions)
    
    # Initialize simulated demand
    simulated_demand = hourly_array + delta_demand
    
    # Distribute the reduced peak demand to off-peak hours
    off_peak_mask = np.array([1 if h in off_peak_hours else 0 for h in range(24)])
    off_peak_sum = np.sum(hourly_array * off_peak_mask)
    
    if off_peak_sum > 0:
        simulated_demand += total_reduced_demand * (hourly_array * off_peak_mask) / off_peak_sum
        
    # Ensure no negative demand
    simulated_demand = np.maximum(simulated_demand, 0.0)
    
    # Re-normalize to strictly conserve total demand
    new_total = np.sum(simulated_demand)
    if new_total > 0:
        simulated_demand = simulated_demand * (original_total / new_total)
        
    return simulated_demand, price_change


def calculate_topsis_rankings(data, weights=None):
    """
    TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
    다중 기준 의사결정(MCDA) 알고리즘을 사용하여 최적 충전소 입지 우선순위를 도출합니다.
    """
    if data.empty:
        return data
        
    df = data.copy()
    
    if weights is None:
        weights = {
            "전력_부하지수": 0.35,
            "인프라_부하지수": 0.35,
            "충전소_밀집도_역수": 0.15,
            "전력망_완화율": 0.15
        }
        
    # 1. 평가 피처 행렬 계산
    # 충전소 밀집도 역수 = 1 / (충전소개수 + 1)
    if "충전소개수" in df.columns:
        df["충전소_밀집도_역수"] = 1.0 / (df["충전소개수"] + 1.0)
    else:
        df["충전소_밀집도_역수"] = 1.0 / (df["전체_충전기대수"] * 0.1 + 1.0)
        
    # 설치 비용 대비 전력망 완화율 = 총용량_kW / (총_전력판매량 + 1.0)
    if "총용량_kW" in df.columns and "총_전력판매량" in df.columns:
        df["전력망_완화율"] = df["총용량_kW"] / (df["총_전력판매량"] + 1.0)
    else:
        df["전력망_완화율"] = 1.0 / (df["전력_부하지수"] + 1.0)
        
    features = list(weights.keys())
    
    # 2. 의사결정 행렬 (Decision Matrix) 생성
    X = df[features].values.astype(float)
    
    # 3. 정규화 (Vector Normalization)
    norm = np.sqrt(np.sum(X**2, axis=0))
    norm = np.where(norm == 0, 1e-5, norm)
    R = X / norm
    
    # 4. 가중치 정규화 의사결정 행렬
    w = np.array([weights[f] for f in features])
    w = w / np.sum(w)
    V = R * w
    
    # 5. 양의 이상적 해(PIS)와 음의 이상적 해(NIS) 결정
    pis = np.max(V, axis=0)
    nis = np.min(V, axis=0)
    
    # 6. Separation Measures 계산
    S_plus = np.sqrt(np.sum((V - pis)**2, axis=1))
    S_minus = np.sqrt(np.sum((V - nis)**2, axis=1))
    
    # 7. relative closeness score (C_i) 계산
    denom = S_plus + S_minus
    denom = np.where(denom == 0, 1e-5, denom)
    closeness = S_minus / denom
    
    df["TOPSIS_점수"] = closeness
    df["TOPSIS_순위"] = df["TOPSIS_점수"].rank(ascending=False, method="min").astype(int)
    
    return df


