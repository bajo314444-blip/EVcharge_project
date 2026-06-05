import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import json
from sklearn.neighbors import BallTree

DEFAULT_DATA_DIR = Path(r"./dataset")

SEOUL_FILE = "서울 전기차 충전소 설치현황.xlsx"
GYEONGGI_FILE = "경기도 전기차 충전소 설치현황.xlsx"
SALES_FILE = "전국 전기차 충전 전력 판매실적.csv"
REGISTER_FILE = "한국교통안전공단_전국_전기차_차종별_용도별_차량_등록대수(운행차량기준)_20250407.csv"
HOURLY_LOAD_FILE = "한국전력공사_전기차 시간대별 충전부하_20240930.csv"
LOCATION_FILE = "한국전력공사_전기차충전소위경도_20250531.csv"
CAPACITY_FILE = "한국전력공사_충전소별 충전기 용량 정보_20240603.csv"
MONTHLY_PUBLIC_FAST_FILE = (
    "환경부 공공급속 충전기 연월별 충전량, 충전횟수, 충전시간(2015년 ~ 2025년8월)_"
    "년월별_20250923_FF.csv"
)

METRO_LONG = ["서울특별시", "경기도", "인천광역시"]
METRO_SHORT = ["서울", "경기", "인천"]
SIDO_MAP = {
    "서울특별시": "서울",
    "서울": "서울",
    "경기도": "경기",
    "경기": "경기",
    "인천광역시": "인천",
    "인천": "인천",
}

def normalize_sido(value):
    if pd.isna(value):
        return ""
    return SIDO_MAP.get(str(value).strip(), str(value).strip())

def read_csv_safely(path, **kwargs):
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, **kwargs)

def get_city_from_address(value):
    parts = str(value).strip().split()
    return parts[1] if len(parts) > 1 else ""

def get_sido_from_address(value):
    parts = str(value).strip().split()
    return normalize_sido(parts[0] if parts else "")

@st.cache_data(show_spinner="데이터를 불러오고 전처리하는 중입니다...")
def load_all_data(data_dir_text):
    data_dir = Path(data_dir_text)
    required = [
        SEOUL_FILE,
        GYEONGGI_FILE,
        SALES_FILE,
        REGISTER_FILE,
        LOCATION_FILE,
        CAPACITY_FILE,
        MONTHLY_PUBLIC_FAST_FILE,
    ]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError("데이터 폴더에서 찾지 못한 파일: " + ", ".join(missing))

    seoul_built = pd.read_excel(data_dir / SEOUL_FILE)
    gyeonggi_built = pd.read_excel(data_dir / GYEONGGI_FILE)
    sales = read_csv_safely(data_dir / SALES_FILE)
    register = read_csv_safely(data_dir / REGISTER_FILE)
    location = read_csv_safely(data_dir / LOCATION_FILE)
    capacity = read_csv_safely(data_dir / CAPACITY_FILE)

    sales["시도"] = sales["시도"].map(normalize_sido)
    sales = sales[sales["시도"].isin(METRO_SHORT)].copy()
    sales["용도"] = sales["충전요금"].replace({"사업용": "사업자용"})
    sales["시군구"] = sales["시군구"].astype(str).str.split().str[0]
    sales["판매량"] = pd.to_numeric(sales["판매량"], errors="coerce").fillna(0)
    sales["판매수입"] = pd.to_numeric(sales["판매수입"], errors="coerce").fillna(0)
    demand_sales = (
        sales.groupby(["시도", "시군구", "용도"], as_index=False)
        .agg(총_전력판매량=("판매량", "sum"), 총_판매수입=("판매수입", "sum"))
    )

    register = register[register["시군구별"].astype(str).str.contains("서울|경기|인천", na=False)].copy()
    register["시도"] = register["시군구별"].apply(lambda x: normalize_sido(str(x).split()[0]))
    register["시군구"] = register["시군구별"].apply(get_city_from_address)
    register["용도"] = register["용도별"].replace({"비사업용": "자가용", "사업용": "사업자용"})
    register["계"] = pd.to_numeric(register["계"], errors="coerce").fillna(0)
    demand_ev = (
        register.groupby(["시도", "시군구", "용도"], as_index=False)
        .agg(전기차_전체대수=("계", "sum"))
    )
    demand = pd.merge(demand_ev, demand_sales, on=["시도", "시군구", "용도"], how="outer").fillna(0)

    built = pd.concat([seoul_built, gyeonggi_built], ignore_index=True)
    built["시도"] = built["주소"].apply(get_sido_from_address)
    built["시군구"] = built["주소"].apply(get_city_from_address)
    for col in ["급속충전기(대)", "완속충전기(대)"]:
        built[col] = pd.to_numeric(built[col], errors="coerce").fillna(0)
    built_summary = (
        built.groupby(["시도", "시군구"], as_index=False)
        .agg(급속충전기_대수=("급속충전기(대)", "sum"), 완속충전기_대수=("완속충전기(대)", "sum"))
    )

    location = location[location["충전소주소"].astype(str).str.contains("|".join(METRO_LONG), na=False)].copy()
    location["시도"] = location["충전소주소"].apply(get_sido_from_address)
    location["시군구"] = location["충전소주소"].apply(get_city_from_address)
    capacity["충전기용량(kw)"] = pd.to_numeric(capacity["충전기용량(kw)"], errors="coerce").fillna(0)

    if "충전소ID" in capacity.columns and "충전소ID" in location.columns:
        supply = pd.merge(
            capacity,
            location[["충전소ID", "충전소명", "충전소주소", "시도", "시군구", "위도", "경도"]],
            on="충전소ID",
            how="right",
            suffixes=("_용량", ""),
        )
        supply["충전소명"] = supply["충전소명"].fillna(supply.get("충전소명_용량", ""))
    else:
        supply = pd.merge(
            capacity,
            location[["충전소명", "충전소주소", "시도", "시군구", "위도", "경도"]],
            on="충전소명",
            how="right",
        )

    supply = supply.dropna(subset=["위도", "경도"]).copy()
    supply_summary = (
        supply.groupby(["시도", "시군구"], as_index=False)
        .agg(
            충전소개수=("충전소명", "nunique"),
            충전기대수=("충전기용량(kw)", "count"),
            총용량_kW=("충전기용량(kw)", "sum"),
            위도=("위도", "mean"),
            경도=("경도", "mean"),
        )
    )

    supply_total = pd.merge(built_summary, supply_summary, on=["시도", "시군구"], how="outer")
    final = pd.merge(demand, supply_total, on=["시도", "시군구"], how="inner")
    
    num_cols = ["전기차_전체대수", "총_전력판매량", "총_판매수입", "급속충전기_대수", "완속충전기_대수", "충전소개수", "충전기대수", "총용량_kW"]
    # Convert appropriate 0s or missing values to NaN for imputation, or just impute existing NaNs from merges
    imputer = IterativeImputer(random_state=42, max_iter=10)
    final[num_cols] = imputer.fit_transform(final[num_cols])
    # Ensure no negative values after imputation
    final[num_cols] = final[num_cols].clip(lower=0)

    final["전체_충전기대수"] = final["급속충전기_대수"] + final["완속충전기_대수"]
    final["전체_충전기대수"] = final["전체_충전기대수"].where(final["전체_충전기대수"] > 0, final["충전기대수"])
    final = final[(final["전체_충전기대수"] > 0) & (final["총용량_kW"] > 0)].copy()
    final["인프라_부하지수"] = final["전기차_전체대수"] / final["전체_충전기대수"]
    final["전력_부하지수"] = final["총_전력판매량"] / final["총용량_kW"]
    final["전력_부하지수"] = final["전력_부하지수"].replace([np.inf, -np.inf], np.nan).fillna(0)
    final["인프라_부하지수"] = final["인프라_부하지수"].replace([np.inf, -np.inf], np.nan).fillna(0)
    final["지역"] = final["시도"] + " " + final["시군구"]
    
    # 2.5단계: 피처 엔지니어링 (다중공선성 제거 및 파생 변수 생성)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    infra_cols = ["급속충전기_대수", "완속충전기_대수", "총용량_kW"]
    pca = PCA(n_components=1, random_state=42)
    infra_scaled = StandardScaler().fit_transform(final[infra_cols])
    pca_raw = pca.fit_transform(infra_scaled)[:, 0]
    # PCA 특성상 평균이 0이 되고 음수가 발생하므로, 직관적인 해석을 위해 0~100점 척도의 지수(Index)로 변환합니다.
    final["충전인프라_규모_PCA"] = MinMaxScaler(feature_range=(0, 100)).fit_transform(pca_raw.reshape(-1, 1)).flatten()
    
    final["충전기_1대당_평균용량"] = final["총용량_kW"] / final["전체_충전기대수"]
    final["충전기_1대당_평균용량"] = final["충전기_1대당_평균용량"].replace([np.inf, -np.inf], np.nan).fillna(0)

    monthly_cols = [
        "시도",
        "군구",
        "대분류",
        "소분류",
        "충전소명",
        "충전소ID",
        "연",
        "월",
        "충전량_합산",
        "충전횟수_합산",
        "충전시간_합산",
    ]
    monthly = read_csv_safely(data_dir / MONTHLY_PUBLIC_FAST_FILE, usecols=monthly_cols)
    monthly = monthly[monthly["시도"].isin(METRO_LONG)].copy()
    monthly["시도"] = monthly["시도"].map(normalize_sido)
    monthly = monthly.rename(columns={"군구": "시군구"})
    monthly["연월"] = pd.to_datetime(
        monthly["연"].astype(str) + "-" + monthly["월"].astype(str).str.zfill(2) + "-01"
    )
    for col in ["충전량_합산", "충전횟수_합산", "충전시간_합산"]:
        monthly[col] = pd.to_numeric(monthly[col], errors="coerce").fillna(0)

    monthly_region = (
        monthly.groupby(["연월", "시도", "시군구"], as_index=False)
        .agg(
            월_충전량=("충전량_합산", "sum"),
            월_충전횟수=("충전횟수_합산", "sum"),
            월_충전시간=("충전시간_합산", "sum"),
            운영_충전소수=("충전소ID", "nunique"),
        )
    )
    monthly_region = pd.merge(
        monthly_region,
        supply_total[["시도", "시군구", "총용량_kW", "위도", "경도"]],
        on=["시도", "시군구"],
        how="left",
    )
    monthly_region["월별_부하지수"] = monthly_region["월_충전량"] / monthly_region["총용량_kW"].replace(0, np.nan)
    monthly_region["월별_부하지수"] = monthly_region["월별_부하지수"].replace([np.inf, -np.inf], np.nan)
    monthly_region["지역"] = monthly_region["시도"] + " " + monthly_region["시군구"]

    hourly = read_csv_safely(data_dir / HOURLY_LOAD_FILE) if (data_dir / HOURLY_LOAD_FILE).exists() else pd.DataFrame()
    return final, monthly_region, hourly


def load_highway_data(data_dir_text):
    data_dir = Path(data_dir_text)
    hw_dir = data_dir / "highway_adress"
    if not hw_dir.exists():
        return pd.DataFrame()
        
    highway_nodes = []
    for fpath in hw_dir.glob("etc_page*.json"):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data.get("list", []):
                    if item.get("xValue") and item.get("yValue"):
                        highway_nodes.append({
                            "unitName": str(item["unitName"]).strip(),
                            "routeName": str(item.get("routeName", "알수없음")).strip(),
                            "경도": float(item["xValue"]),
                            "위도": float(item["yValue"])
                        })
        except Exception:
            pass
            
    if not highway_nodes:
        return pd.DataFrame()
        
    hw_df = pd.DataFrame(highway_nodes).dropna(subset=["위도", "경도"])
    # Some names overlap in upbound/downbound, keep them distinct by coordinate
    hw_df = hw_df.drop_duplicates(subset=["위도", "경도"])
    
    # Load charging stations to match
    location = read_csv_safely(data_dir / LOCATION_FILE)
    capacity = read_csv_safely(data_dir / CAPACITY_FILE)
    
    location["위도"] = pd.to_numeric(location["위도"], errors="coerce")
    location["경도"] = pd.to_numeric(location["경도"], errors="coerce")
    capacity["충전기용량(kw)"] = pd.to_numeric(capacity["충전기용량(kw)"], errors="coerce").fillna(0)
    
    # Merge supply
    if "충전소ID" in capacity.columns and "충전소ID" in location.columns:
        supply = pd.merge(capacity, location[["충전소ID", "충전소명", "위도", "경도"]], on="충전소ID", how="inner")
    else:
        supply = pd.merge(capacity, location[["충전소명", "위도", "경도"]], on="충전소명", how="inner")
        
    supply = supply.dropna(subset=["위도", "경도"]).copy()
    
    # BallTree Spatial Join
    hw_coords = np.radians(hw_df[["위도", "경도"]].values)
    supply_coords = np.radians(supply[["위도", "경도"]].values)
    
    tree = BallTree(supply_coords, metric='haversine')
    
    # Query within 3km (3 / 6371 radians)
    radius_km = 3.0
    radius_rad = radius_km / 6371.0
    
    indices, _ = tree.query_radius(hw_coords, r=radius_rad, return_distance=True)
    
    hw_capacity = []
    hw_chargers = []
    
    for idx_list in indices:
        if len(idx_list) > 0:
            matched_supply = supply.iloc[idx_list]
            hw_capacity.append(matched_supply["충전기용량(kw)"].sum())
            hw_chargers.append(len(matched_supply))
        else:
            hw_capacity.append(0.0)
            hw_chargers.append(0)
            
    hw_df["총용량_kW"] = hw_capacity
    hw_df["충전기대수"] = hw_chargers
    
    # Max Capacity 제약: 기본 휴게소당 최소 5대, 최대 20대로 가정, 기존 대수가 많으면 더 줌
    hw_df["Max_Capacity"] = np.clip(hw_df["충전기대수"] * 2, 5, 20)
    
    return hw_df

def create_time_spike_features(dates):
    """
    datetime 리스트를 받아 Time Spike 4대 변수를 생성하여 DataFrame으로 반환
    """
    df = pd.DataFrame({"datetime": pd.to_datetime(dates)})
    
    # 1. is_commute_time
    is_weekday = df["datetime"].dt.dayofweek < 5
    is_morning_commute = df["datetime"].dt.hour.between(7, 8) # 7:00 ~ 8:59
    is_evening_commute = df["datetime"].dt.hour.between(17, 18) # 17:00 ~ 18:59
    df["is_commute_time"] = (is_weekday & (is_morning_commute | is_evening_commute)).astype(int)
    
    # 4. is_weekend
    df["is_weekend"] = (~is_weekday).astype(int)
    
    # 2. is_holiday & 3. is_golden_week (simplified heuristic based on custom dictionary)
    # 실제로는 pytimekr 등을 쓰지만, 프로토타입을 위해 하드코딩된 주요 공휴일 매핑
    holidays_2024 = [
        "2024-01-01", "2024-02-09", "2024-02-10", "2024-02-11", "2024-02-12",
        "2024-03-01", "2024-04-10", "2024-05-05", "2024-05-06", "2024-05-15",
        "2024-06-06", "2024-08-15", "2024-09-16", "2024-09-17", "2024-09-18",
        "2024-10-03", "2024-10-09", "2024-12-25"
    ]
    golden_weeks = [
        # 추석 연휴 (주말 포함)
        "2024-09-14", "2024-09-15", "2024-09-16", "2024-09-17", "2024-09-18",
        # 5월 징검다리
        "2024-05-04", "2024-05-05", "2024-05-06"
    ]
    
    date_str = df["datetime"].dt.strftime("%Y-%m-%d")
    df["is_holiday"] = date_str.isin(holidays_2024).astype(int)
    df["is_golden_week"] = date_str.isin(golden_weeks).astype(int)
    
    return df


@st.cache_data(show_spinner="부트스트랩 CI 산출 중...")
def cached_bootstrap(best_name, _model, X_test, y_test):
    from utils.optimization import calculate_bootstrap_ci
    return calculate_bootstrap_ci(_model, X_test, y_test, n_iterations=100)


@st.cache_data(show_spinner="적대적 공격 방어력 평가 중...")
def cached_adversarial(best_name, _model, X_test, y_test):
    from utils.optimization import run_adversarial_attack
    return run_adversarial_attack(_model, X_test, y_test)


@st.cache_data(show_spinner="피처 민감도 분석 중...")
def cached_ablation(best_name, _model, X_train, y_train, X_test, y_test, importances):
    from utils.optimization import run_ablation_study
    return run_ablation_study(_model, X_train, y_train, X_test, y_test, importances)


@st.cache_data(show_spinner="DCA 곡선 산출 중...")
def cached_dca(best_name, _model, X_test, y_test):
    from utils.optimization import calculate_dca
    return calculate_dca(_model, X_test, y_test)


@st.cache_data(show_spinner="중첩 10-겹 교차검증 (Nested 10-fold CV) 수행 중... (시간이 걸릴 수 있습니다)")
def cached_nested_cv(best_name, _model, X, y):
    from utils.optimization import run_nested_cv
    return run_nested_cv(_model, X, y)


@st.cache_data(show_spinner="공간적 외부 검증(Spatial External Validation) 수행 중...")
def cached_spatial_external_validation(best_name, _model, X, y, holdout_region):
    from utils.optimization import run_spatial_external_validation
    return run_spatial_external_validation(_model, X, y, holdout_region)


@st.cache_data(show_spinner="생존 분석 시뮬레이션 중...")
def cached_survival(final_json, growth_rate):
    from utils.optimization import run_survival_simulation
    from io import StringIO
    final_df = pd.read_json(StringIO(final_json), orient="split")
    return run_survival_simulation(final_df, growth_rate)


@st.cache_data(show_spinner="Partial Dependence 계산 중...")
def cached_partial_dependence(best_name, _model, X_all, selected_feature):
    grid = np.linspace(X_all[selected_feature].quantile(0.05), X_all[selected_feature].quantile(0.95), 30)
    pd_rows = []
    for value in grid:
        X_tmp = X_all.copy()
        X_tmp[selected_feature] = value
        pd_rows.append({"Feature value": value, "Mean prediction": _model.predict(X_tmp).mean()})
    return pd.DataFrame(pd_rows)


class ONNXModelWrapper:
    def __init__(self, onnx_path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, X):
        import numpy as np
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)
            
        if len(X_arr.shape) == 1:
            X_arr = X_arr.reshape(1, -1)
            
        preds = self.session.run([self.output_name], {self.input_name: X_arr})[0]
        if len(preds.shape) > 1 and preds.shape[1] == 1:
            preds = preds.squeeze(axis=1)
        return preds


def load_precomputed_analytics(json_path, onnx_path):
    import json
    import os
    import pandas as pd
    import numpy as np
    
    if not os.path.exists(json_path) or not os.path.exists(onnx_path):
        raise FileNotFoundError("ONNX model or JSON precomputed analytics file not found.")
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    metrics = pd.DataFrame(data["metrics"])
    importance = pd.DataFrame(data["importance"])
    predictions = pd.DataFrame(data["predictions"])
    
    X = pd.DataFrame(data["X"])
    X_test = pd.DataFrame(data["X_test"])
    X_train = pd.DataFrame(data["X_train"])
    
    y = pd.Series(data["y"])
    y_test = pd.Series(data["y_test"])
    y_train = pd.Series(data["y_train"])
    
    best_name = data["best_name"]
    best_model = ONNXModelWrapper(onnx_path)
    
    model_state = {
        "metrics": metrics,
        "importance": importance,
        "predictions": predictions,
        "best_name": best_name,
        "feature_columns": data["feature_columns"],
        "model_groups": data["model_groups"],
        "X": X,
        "y": y,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "models": {best_name: best_model},
    }
    
    if "precomputed_tsne_xy" in data:
        model_state["precomputed_tsne_xy"] = np.array(data["precomputed_tsne_xy"])
    if "precomputed_umap_xy" in data:
        model_state["precomputed_umap_xy"] = np.array(data["precomputed_umap_xy"])
        model_state["precomputed_umap_title"] = data.get("precomputed_umap_title", "UMAP")
    if "precomputed_cca_x_c" in data and "precomputed_cca_y_c" in data:
        model_state["precomputed_cca_x_c"] = np.array(data["precomputed_cca_x_c"])
        model_state["precomputed_cca_y_c"] = np.array(data["precomputed_cca_y_c"])
        
    if "precomputed_bootstrap" in data:
        pb = data["precomputed_bootstrap"]
        model_state["precomputed_bootstrap"] = (
            pb["ci_rmse"], 
            pb["ci_r2"], 
            np.array(pb["bootstrap_rmse"]), 
            np.array(pb["bootstrap_r2"])
        )
        
    if "precomputed_nested_cv" in data:
        pnc = data["precomputed_nested_cv"]
        model_state["precomputed_nested_cv"] = (
            pnc["mean_rmse"],
            pnc["std_rmse"],
            np.array(pnc["outer_scores"])
        )
        
    if "precomputed_adversarial" in data:
        model_state["precomputed_adversarial"] = pd.DataFrame(data["precomputed_adversarial"])
        
    if "precomputed_ablation" in data:
        model_state["precomputed_ablation"] = pd.DataFrame(data["precomputed_ablation"])
        
    if "precomputed_dca" in data:
        model_state["precomputed_dca"] = pd.DataFrame(data["precomputed_dca"])
        
    if "precomputed_spatial" in data:
        ps = data["precomputed_spatial"]
        spatial_decoded = {}
        for region, metrics_list in ps.items():
            spatial_decoded[region] = (
                metrics_list[0],
                metrics_list[1],
                metrics_list[2],
                None
            )
        model_state["precomputed_spatial"] = spatial_decoded
        
    if "precomputed_survival_5" in data:
        model_state["precomputed_survival_5"] = pd.DataFrame(data["precomputed_survival_5"])
        
    if "precomputed_roc_data" in data:
        model_state["precomputed_roc_data"] = data["precomputed_roc_data"]
        
    model_state_smote = None
    if "model_state_smote_metrics" in data:
        smote_metrics = pd.DataFrame(data["model_state_smote_metrics"])
        model_state_smote = {
            "metrics": smote_metrics,
            "best_name": data.get("model_state_smote_best_name", "RandomForest (Tuned)")
        }
        
    return model_state, model_state_smote



