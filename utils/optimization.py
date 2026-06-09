# ============================================================
# 파일명: optimization.py
# 설명: 전기차 충전 인프라 최적 배치, 모델 강건성 평가,
#       동적 가격 시뮬레이션, TOPSIS 다중 기준 의사결정 알고리즘 모듈
# ============================================================

import numpy as np  # numpy(넘파이) 수치 연산 라이브러리를 np로 import(임포트)
import pandas as pd  # pandas(판다스) 데이터 분석 라이브러리를 pd로 import(임포트)
from scipy.optimize import linprog  # scipy(사이파이)의 linprog(선형 계획법) 최적화 함수를 import(임포트)

# --- 고속도로 충전기 최적 배치 함수 (LP 기반) ---
def optimize_highway_chargers(hw_df, budget):  # 고속도로 휴게소/IC 최적 충전기 배치 함수 정의: 데이터와 예산을 받음
    """
    고속도로 휴게소/IC 최적 입지 선정 알고리즘 (Linear Programming 기반)
    주어진 예산(budget) 내에서 부하 감소 효율성이 가장 높은 노드 조합을 찾습니다.
    
    LP Formulation:
    Maximize sum( Delta_L_i * X_i )  => Minimize sum( -Delta_L_i * X_i )
    Subject to:
        sum(X_i) <= budget
        0 <= X_i <= Max_Capacity_i
    """
    n_nodes = len(hw_df)  # 전체 노드(node) 수 (고속도로 거점 수) 계산
    if n_nodes == 0 or budget <= 0:  # 노드가 없거나 예산이 0 이하인 경우 처리
        hw_df = hw_df.copy()  # 원본 보호를 위해 DataFrame(데이터프레임) 복사
        hw_df["최적_추가대수"] = 0  # 최적 추가 대수 column(컬럼)을 0으로 초기화
        hw_df["최적화후_부하점수"] = hw_df.get("부하_예측점수", 0)  # 최적화 후 부하 점수를 기존 예측 점수로 설정
        return hw_df  # 결과 DataFrame(데이터프레임) 반환(리턴)

    # --- 부하 예측 점수가 없을 경우 초기화 에러 방지 ---
    if "부하_예측점수" not in hw_df.columns:  # 부하_예측점수 column(컬럼)이 존재하지 않는 경우
        hw_df["부하_예측점수"] = np.random.uniform(30, 90, n_nodes)  # 30~90 범위의 랜덤 값으로 임의 점수 생성

    # --- 각 노드별 충전기 1대 추가 시의 부하 감소량(효율성) 계산 ---
    # 기존 점수의 약 2~8%가 감소한다고 가정 (노드 용량별 차등)
    delta_L = hw_df["부하_예측점수"].values * np.random.uniform(0.02, 0.08, n_nodes)  # 각 노드의 부하 감소 효율성(delta_L) 계산

    # --- LP Solver(선형 계획법 솔버) 적용: scipy.optimize.linprog 사용 ---
    c = -delta_L  # linprog는 minimize(최소화)를 수행하므로 목적함수를 음수로 변환

    A_ub = np.ones((1, n_nodes))  # 부등식 제약 행렬: 모든 노드의 설치 대수 합
    b_ub = np.array([budget])  # 부등식 제약 상한: 총 예산(budget)

    # --- 각 노드별 Max_Capacity(최대 설치 용량) upper bound(상한) 제약 ---
    bounds = [(0, cap) for cap in hw_df["Max_Capacity"]]  # 각 노드의 설치 가능 범위를 (0, 최대용량)으로 설정

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')  # HiGHS 솔버로 선형 계획 문제(LP) 풀기

    hw_df = hw_df.copy()  # 원본 보호를 위해 DataFrame(데이터프레임) 복사
    if res.success:  # 최적화가 성공적으로 수렴한 경우
        hw_df["최적_추가대수"] = np.round(res.x).astype(int)  # 최적화 결과를 정수로 반올림하여 저장
    else:  # 최적화가 실패한 경우
        hw_df["최적_추가대수"] = 0  # 최적 추가 대수를 0으로 설정

    # --- 최적화 후 부하 점수 갱신 ---
    hw_df["최적화후_부하점수"] = np.clip(  # 부하 점수를 0~100 범위로 clip(클리핑)
        hw_df["부하_예측점수"] - (delta_L * hw_df["최적_추가대수"]),  # 기존 부하 점수에서 감소량을 차감
        0, 100  # 최소값 0, 최대값 100으로 범위 제한
    )

    return hw_df  # 최적화 결과가 포함된 DataFrame(데이터프레임) 반환(리턴)


# ==========================================
# Robustness(강건성) & Model Evaluation(모델 평가) Algorithms(알고리즘)
# ==========================================
from sklearn.model_selection import KFold  # KFold(K-폴드) 교차검증 클래스를 import(임포트)
from sklearn.metrics import mean_squared_error, r2_score  # MSE(평균 제곱 오차)와 R²(결정계수) 평가 함수를 import(임포트)
import copy  # copy(복사) 모듈을 import(임포트): 모델 깊은 복사에 사용

# --- Bootstrap(부트스트랩) 95% 신뢰구간 계산 함수 ---
def calculate_bootstrap_ci(best_model, X_test, y_test, n_iterations=100):  # Bootstrap(부트스트랩) 신뢰구간 계산 함수 정의
    """
    Bootstrap 95% Confidence Interval
    """
    rmse_scores = []  # RMSE(평균 제곱근 오차) 점수를 저장할 리스트 초기화
    r2_scores = []  # R²(결정계수) 점수를 저장할 리스트 초기화

    n_size = len(y_test)  # 테스트 세트 샘플 수 계산
    X_test_arr = np.array(X_test)  # X_test를 numpy array(배열)로 변환
    y_test_arr = np.array(y_test)  # y_test를 numpy array(배열)로 변환

    # --- Bootstrap(부트스트랩) 반복 샘플링 ---
    for i in range(n_iterations):  # n_iterations(반복 횟수)만큼 Bootstrap(부트스트랩) 수행
        np.random.seed(42 + i)  # 재현성을 위한 난수 seed(시드) 설정 (반복마다 다른 시드)
        indices = np.random.choice(range(n_size), size=n_size, replace=True)  # 복원 추출(replacement)로 Bootstrap(부트스트랩) 샘플 인덱스 생성
        X_sample = X_test_arr[indices]  # Bootstrap(부트스트랩) 샘플의 입력 데이터 추출
        y_sample = y_test_arr[indices]  # Bootstrap(부트스트랩) 샘플의 타겟 데이터 추출

        preds = best_model.predict(X_sample)  # 최적 모델로 Bootstrap(부트스트랩) 샘플 예측 수행
        rmse = np.sqrt(mean_squared_error(y_sample, preds))  # RMSE(평균 제곱근 오차) 계산
        r2 = r2_score(y_sample, preds)  # R²(결정계수) 계산

        rmse_scores.append(rmse)  # RMSE 점수를 리스트에 추가
        r2_scores.append(r2)  # R² 점수를 리스트에 추가

    ci_rmse = (np.percentile(rmse_scores, 2.5), np.percentile(rmse_scores, 97.5))  # RMSE의 95% 신뢰구간 (2.5%, 97.5% 분위수) 계산
    ci_r2 = (np.percentile(r2_scores, 2.5), np.percentile(r2_scores, 97.5))  # R²의 95% 신뢰구간 (2.5%, 97.5% 분위수) 계산

    return ci_rmse, ci_r2, rmse_scores, r2_scores  # 신뢰구간과 전체 점수 리스트를 반환(리턴)


# --- Adversarial Attack(적대적 공격) 분석 함수: Gaussian Noise(가우시안 노이즈) 주입 ---
def run_adversarial_attack(best_model, X_test, y_test, noise_levels=[0.05, 0.1, 0.2]):  # 적대적 노이즈 공격 분석 함수 정의
    """
    Adversarial Attack Analysis: Inject Gaussian Noise
    """
    results = []  # 결과를 저장할 리스트 초기화
    base_preds = best_model.predict(X_test)  # 원본 데이터에 대한 기본 예측값 계산
    base_rmse = np.sqrt(mean_squared_error(y_test, base_preds))  # 기본(baseline) RMSE(평균 제곱근 오차) 계산

    results.append({"Noise_Level": "0% (Base)", "RMSE": base_rmse, "Drop_Ratio(%)": 0.0})  # 기본(노이즈 없음) 결과를 리스트에 추가

    X_test_arr = np.array(X_test)  # X_test를 numpy array(배열)로 변환
    stds = np.std(X_test_arr, axis=0)  # 각 feature(특성)별 표준편차 계산

    # --- 각 노이즈 수준(noise level)에 대해 공격 수행 ---
    for noise in noise_levels:  # 각 노이즈 비율을 순회
        np.random.seed(42)  # 재현성을 위한 난수 seed(시드) 설정
        noise_matrix = np.random.normal(0, stds * noise, size=X_test_arr.shape)  # feature(특성)별 표준편차에 비례하는 Gaussian noise(가우시안 노이즈) 행렬 생성
        X_adv = X_test_arr + noise_matrix  # 원본 데이터에 노이즈를 추가하여 adversarial(적대적) 데이터 생성

        preds = best_model.predict(X_adv)  # adversarial(적대적) 데이터에 대해 예측 수행
        rmse = np.sqrt(mean_squared_error(y_test, preds))  # 노이즈 적용 후 RMSE(평균 제곱근 오차) 계산
        drop_ratio = (rmse - base_rmse) / base_rmse * 100  # 기본 RMSE 대비 성능 저하율(%) 계산

        results.append({"Noise_Level": f"{int(noise*100)}%", "RMSE": rmse, "Drop_Ratio(%)": drop_ratio})  # 결과를 리스트에 추가

    return pd.DataFrame(results)  # 결과를 DataFrame(데이터프레임)으로 변환하여 반환(리턴)


# --- Ablation Study(절제 연구): Feature(특성) 중요도 민감도 분석 ---
def run_ablation_study(best_model, X_train, y_train, X_test, y_test, importances):  # Ablation Study(절제 연구) 함수 정의: 중요도 기반 feature(특성) 제거 분석
    """
    Feature importance ablation study (Sensitivity analysis)
    """
    from sklearn.base import clone  # sklearn(사이킷런)의 clone(복제) 함수 import(임포트)
    features_ordered = importances["Feature"].tolist()  # 중요도 순서로 정렬된 feature(특성) 이름 리스트 추출

    results = []  # 결과를 저장할 리스트 초기화
    current_features = features_ordered.copy()  # 현재 사용 중인 feature(특성) 리스트 (복사본)

    # --- 가장 덜 중요한 feature(특성)부터 하나씩 제거하며 반복 ---
    for i in range(len(features_ordered) - 1):  # feature(특성) 수 - 1 만큼 반복 (최소 1개는 유지)
        if len(current_features) < 2:  # 남은 feature(특성)가 2개 미만이면 중단
            break  # 반복 종료

        try:  # sklearn clone(복제) 시도
            model_clone = clone(best_model)  # 최적 모델의 하이퍼파라미터를 복제한 새 모델 생성
        except:  # clone(복제) 실패 시 (커스텀 모델 등)
            model_clone = copy.deepcopy(best_model)  # deepcopy(깊은 복사)로 모델 복제

        model_clone.fit(X_train[current_features], y_train)  # 현재 남은 feature(특성)들로만 모델 학습
        preds = model_clone.predict(X_test[current_features])  # 현재 남은 feature(특성)들로 예측 수행
        rmse = np.sqrt(mean_squared_error(y_test, preds))  # RMSE(평균 제곱근 오차) 계산

        results.append({  # 결과를 딕셔너리(dictionary)로 생성하여 리스트에 추가
            "Num_Features": len(current_features),  # 현재 사용 중인 feature(특성) 개수
            "RMSE": rmse,  # 해당 feature(특성) 조합의 RMSE(평균 제곱근 오차)
            "Removed_Feature": features_ordered[-i] if i > 0 else "None"  # 이번 단계에서 제거된 feature(특성) 이름
        })

        current_features.pop()  # 가장 덜 중요한(마지막) feature(특성)를 리스트에서 제거

    return pd.DataFrame(results)  # 결과를 DataFrame(데이터프레임)으로 변환하여 반환(리턴)


# --- DCA(의사결정 곡선 분석): 회귀 모델 적용 버전 ---
def calculate_dca(best_model, X_test, y_test):  # DCA(Decision Curve Analysis, 의사결정 곡선 분석) 함수 정의
    """
    Decision Curve Analysis adapted for Regression.
    """
    preds_continuous = best_model.predict(X_test)  # 최적 모델로 연속 예측값 계산

    thresholds = np.linspace(np.percentile(y_test, 50), np.percentile(y_test, 90), 20)  # 50~90 백분위수 범위에서 20개의 threshold(임계값) 생성
    net_benefits = []  # Net Benefit(순 편익) 결과를 저장할 리스트 초기화

    # --- 각 threshold(임계값)에 대해 Net Benefit(순 편익) 계산 ---
    for th in thresholds:  # 각 threshold(임계값)를 순회
        y_true_bin = (y_test >= th).astype(int)  # 실제값을 threshold(임계값) 기준으로 이진(binary) 분류
        pred_bin = (preds_continuous >= th).astype(int)  # 예측값을 threshold(임계값) 기준으로 이진(binary) 분류

        tp = np.sum((y_true_bin == 1) & (pred_bin == 1))  # True Positive(진양성) 수 계산
        fp = np.sum((y_true_bin == 0) & (pred_bin == 1))  # False Positive(위양성) 수 계산
        n = len(y_test)  # 전체 샘플 수

        prevalence = np.sum(y_true_bin) / n  # 양성 비율(prevalence, 유병률) 계산
        if prevalence == 0 or prevalence == 1:  # prevalence(유병률)가 0 또는 1이면 계산 불가
            continue  # 다음 threshold(임계값)로 이동

        pt = prevalence  # 확률 threshold(확률 임계값)를 prevalence(유병률)로 설정

        nb_model = (tp / n) - (fp / n) * (pt / (1 - pt))  # 모델의 Net Benefit(순 편익) 계산

        tp_all = np.sum(y_true_bin)  # 전체를 양성으로 예측했을 때의 True Positive(진양성) 수
        fp_all = n - tp_all  # 전체를 양성으로 예측했을 때의 False Positive(위양성) 수
        nb_all = (tp_all / n) - (fp_all / n) * (pt / (1 - pt))  # "Treat All(전원 처치)" 전략의 Net Benefit(순 편익) 계산

        net_benefits.append({"Threshold_Value": th, "Model_NB": nb_model, "Treat_All_NB": nb_all, "Treat_None_NB": 0.0})  # 결과를 리스트에 추가

    return pd.DataFrame(net_benefits)  # 결과를 DataFrame(데이터프레임)으로 변환하여 반환(리턴)


# --- Survival Simulation(생존 시뮬레이션): 전기차 증가에 따른 과부하 도달 시점 예측 ---
def run_survival_simulation(final_data, growth_rate=0.05):  # 생존 시뮬레이션 함수 정의: 최종 데이터와 성장률을 받음
    """
    Simulate Time-to-Overload based on EV growth rate.
    """
    df = final_data.copy()  # 원본 보호를 위해 데이터 복사
    critical_threshold = df["전력_부하지수"].quantile(0.8)  # 상위 20%(80 백분위수)를 과부하 임계값(critical threshold)으로 설정

    years = range(1, 16)  # 시뮬레이션 기간: 최대 15년
    survival_data = []  # 생존 분석 결과를 저장할 리스트 초기화

    # --- 각 지역(행)에 대해 과부하 도달 시점 계산 ---
    for i, row in df.iterrows():  # DataFrame(데이터프레임)의 각 행을 순회
        base_load = row["총_전력판매량"]  # 현재 기준 전력 판매량 (기본 부하)
        capacity = row["총용량_kW"]  # 현재 총 충전 용량 (kW)

        overload_year = 15  # 기본값: 15년 내 과부하 미도달로 초기화
        event = 0  # 이벤트 발생 여부: 0=미발생(censored), 1=과부하 발생

        if capacity > 0:  # 용량이 양수인 경우에만 계산 수행
            current_index = base_load / capacity  # 현재 부하 지수 (부하/용량 비율) 계산
            if current_index >= critical_threshold:  # 현재 시점에서 이미 과부하 상태인 경우
                overload_year = 0  # 과부하 도달 시점을 0년(현재)으로 설정
                event = 1  # 이벤트 발생으로 표시
            else:  # 현재는 과부하가 아닌 경우
                # --- 연도별 전기차 부하 성장 시뮬레이션 ---
                for y in years:  # 1년부터 15년까지 순회
                    simulated_load = base_load * ((1 + growth_rate) ** y)  # 지수 성장 모델로 미래 부하 계산
                    simulated_index = simulated_load / capacity  # 미래 부하 지수 계산
                    if simulated_index >= critical_threshold:  # 과부하 임계값을 초과하는 경우
                        overload_year = y  # 과부하 도달 연도 기록
                        event = 1  # 이벤트 발생으로 표시
                        break  # 과부하 도달 시 반복 종료

        survival_data.append({"Region": row["지역"], "Time_to_Overload": overload_year, "Event": event})  # 지역별 생존 분석 결과를 리스트에 추가

    return pd.DataFrame(survival_data)  # 결과를 DataFrame(데이터프레임)으로 변환하여 반환(리턴)


# --- 단일 지역 부하 궤적(trajectory) 계산 함수: 충전 용량 추가 전후 비교 ---
def calculate_single_region_trajectory(base_load, capacity, growth_rate, added_kw, critical_threshold):  # 단일 지역의 15년간 부하 지수 궤적(trajectory)을 계산하는 함수 정의
    """
    Calculate the Load Index trajectory over 15 years for a single region,
    before and after adding charging capacity.
    """
    trajectory = []  # 궤적(trajectory) 데이터를 저장할 리스트 초기화

    overload_year_before = 15  # 설치 전 과부하 도달 연도 기본값: 15년 (미도달)
    overload_year_after = 15  # 설치 후 과부하 도달 연도 기본값: 15년 (미도달)

    # --- 0년부터 15년까지 연도별 궤적 계산 ---
    for y in range(16):  # 0년(현재)부터 15년까지 순회
        # 1단계: EV(전기차) 부하의 지수 성장 계산
        simulated_load = base_load * ((1 + growth_rate) ** y)  # 연도 y에서의 시뮬레이션된 부하 계산

        # 2단계: 설치 전(Base) 부하 지수 계산
        base_index = simulated_load / capacity if capacity > 0 else 0  # 기존 용량 기준 부하 지수 계산 (0으로 나누기 방지)
        if base_index >= critical_threshold and overload_year_before == 15 and y > 0:  # 과부하 임계값 초과 여부 확인
            if simulated_load / capacity >= critical_threshold:  # 엄격한 재평가 수행 (안전 검증)
                pass  # 조건 확인만 수행 (실제 로직은 아래에서 처리)

        if y == 0 and base_index >= critical_threshold:  # 현재 시점(0년)에서 이미 과부하인 경우
            overload_year_before = 0  # 설치 전 과부하 도달 연도를 0년으로 설정
        elif y > 0 and base_index >= critical_threshold and overload_year_before == 15:  # 미래 시점에서 처음 과부하에 도달한 경우
            overload_year_before = y  # 설치 전 과부하 도달 연도 기록

        # 3단계: 설치 후(After) 부하 지수 계산
        after_capacity = capacity + added_kw  # 충전 용량 추가 후의 총 용량 계산
        after_index = simulated_load / after_capacity if after_capacity > 0 else 0  # 추가 후 용량 기준 부하 지수 계산

        if y == 0 and after_index >= critical_threshold:  # 설치 후에도 현재 시점에서 과부하인 경우
            overload_year_after = 0  # 설치 후 과부하 도달 연도를 0년으로 설정
        elif y > 0 and after_index >= critical_threshold and overload_year_after == 15:  # 설치 후 미래 시점에서 처음 과부하에 도달한 경우
            overload_year_after = y  # 설치 후 과부하 도달 연도 기록

        trajectory.append({  # 설치 전 궤적(trajectory) 데이터를 리스트에 추가
            "Year": y,  # 연도
            "상태": "설치 전",  # 상태: 충전기 설치 전
            "부하지수": base_index  # 해당 연도의 설치 전 부하 지수
        })
        trajectory.append({  # 설치 후 궤적(trajectory) 데이터를 리스트에 추가
            "Year": y,  # 연도
            "상태": "설치 후",  # 상태: 충전기 설치 후
            "부하지수": after_index  # 해당 연도의 설치 후 부하 지수
        })

    return pd.DataFrame(trajectory), overload_year_before, overload_year_after  # 궤적 DataFrame(데이터프레임)과 전후 과부하 도달 연도 반환(리턴)


# --- Nested Cross-Validation(중첩 교차검증) 함수 ---
def run_nested_cv(best_model, X, y, n_splits=10):  # Nested CV(중첩 교차검증) 함수 정의: 최적 모델과 전체 데이터를 받음
    """
    Perform Nested 10-fold Cross-Validation for the best model to rigorously evaluate performance.
    Since best_model is already the best_estimator_, we deepcopy it.
    """
    import copy  # copy(복사) 모듈 import(임포트)
    from sklearn.model_selection import KFold, GridSearchCV  # KFold(K-폴드)와 GridSearchCV(그리드서치) import(임포트)
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # RF(랜덤 포레스트)와 GB(그래디언트 부스팅) import(임포트)
    from sklearn.metrics import mean_squared_error  # MSE(평균 제곱 오차) 평가 함수 import(임포트)
    import numpy as np  # numpy(넘파이) 라이브러리를 np로 import(임포트)

    # --- 커스텀 모델은 sklearn clone(복제) 불가하므로 deepcopy(깊은 복사) 사용 ---
    base_estimator = copy.deepcopy(best_model)  # 최적 모델을 deepcopy(깊은 복사)하여 기본 estimator(추정기)로 사용

    # --- 모델 유형에 따른 하이퍼파라미터 그리드 설정 ---
    if isinstance(base_estimator, RandomForestRegressor):  # RandomForest(랜덤 포레스트) 모델인 경우
        param_grid = {'max_depth': [3, 5, 10], 'min_samples_leaf': [1, 2]}  # RF 전용 하이퍼파라미터 그리드 설정
    elif isinstance(base_estimator, GradientBoostingRegressor):  # GradientBoosting(그래디언트 부스팅) 모델인 경우
        param_grid = {'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]}  # GB 전용 하이퍼파라미터 그리드 설정
    else:  # 기타 모델 (커스텀 모델 등)인 경우
        param_grid = {}  # 하이퍼파라미터 튜닝 없이 진행

    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)  # 외부 CV(교차검증): n_splits 폴드로 설정
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)  # 내부 CV(교차검증): 3-fold로 설정 (하이퍼파라미터 튜닝용)

    outer_scores = []  # 외부 CV(교차검증) RMSE 점수를 저장할 리스트 초기화

    X_arr = np.array(X)  # X를 numpy array(배열)로 변환
    y_arr = np.array(y)  # y를 numpy array(배열)로 변환

    # --- 외부 CV(교차검증) 루프 ---
    for train_idx, test_idx in outer_cv.split(X_arr):  # 외부 CV(교차검증) 폴드를 순회
        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]  # 외부 학습/테스트 데이터 분할
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]  # 외부 학습/테스트 타겟 분할

        if param_grid:  # 하이퍼파라미터 그리드가 있는 경우 (내부 CV 수행)
            grid = GridSearchCV(base_estimator, param_grid, cv=inner_cv, scoring='neg_mean_squared_error', n_jobs=-1)  # 내부 GridSearchCV(그리드서치) 설정: 병렬 실행
            grid.fit(X_tr, y_tr)  # 내부 CV(교차검증)로 최적 하이퍼파라미터 탐색 및 학습
            best_local_model = grid.best_estimator_  # 내부 CV(교차검증)에서 찾은 최적 모델 추출
        else:  # 하이퍼파라미터 그리드가 없는 경우 (단순 학습)
            best_local_model = copy.deepcopy(base_estimator)  # 기본 모델을 deepcopy(깊은 복사)
            best_local_model.fit(X_tr, y_tr)  # 외부 학습 데이터로 직접 학습

        preds = best_local_model.predict(X_te)  # 외부 테스트 데이터에 대해 예측 수행
        rmse = np.sqrt(mean_squared_error(y_te, preds))  # RMSE(평균 제곱근 오차) 계산
        outer_scores.append(rmse)  # 외부 CV(교차검증) RMSE 점수를 리스트에 추가

    return np.mean(outer_scores), np.std(outer_scores), outer_scores  # RMSE 평균, 표준편차, 전체 점수 리스트를 반환(리턴)


# --- Spatial External Validation(공간 외부 검증) 함수: 특정 지역을 홀드아웃하여 공간적 일반화 성능 평가 ---
def run_spatial_external_validation(best_model, X, y, holdout_region="인천"):  # 공간 외부 검증 함수 정의: 특정 지역을 홀드아웃(holdout)
    """
    Perform Spatial External Validation by holding out a specific region completely from training.
    """
    import copy  # copy(복사) 모듈 import(임포트)
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # 평가 지표(MSE, MAE, R²) 함수 import(임포트)
    import numpy as np  # numpy(넘파이) 라이브러리를 np로 import(임포트)

    # --- 선택된 지역에 따른 테스트 mask(마스크) 생성 ---
    if holdout_region == "서울":  # 홀드아웃(holdout) 지역이 서울인 경우
        test_mask = X["region_seoul"] == 1  # 서울 지역 데이터를 테스트 세트로 선택
    elif holdout_region == "경기":  # 홀드아웃(holdout) 지역이 경기인 경우
        test_mask = X["region_gyeonggi"] == 1  # 경기 지역 데이터를 테스트 세트로 선택
    elif holdout_region == "인천":  # 홀드아웃(holdout) 지역이 인천인 경우
        test_mask = X["region_incheon"] == 1  # 인천 지역 데이터를 테스트 세트로 선택
    else:  # 알 수 없는 지역이 입력된 경우
        raise ValueError(f"Unknown region: {holdout_region}")  # ValueError(값 오류) 예외 발생

    train_mask = ~test_mask  # 테스트 mask(마스크)의 반전으로 학습 mask(마스크) 생성

    X_train, y_train = X[train_mask], y[train_mask]  # 학습용 데이터 분할 (홀드아웃 지역 제외)
    X_test, y_test = X[test_mask], y[test_mask]  # 테스트용 데이터 분할 (홀드아웃 지역만)

    if len(X_test) == 0:  # 테스트 데이터가 비어있는 경우
        return None, None, None, "선택한 배제 지역에 해당하는 데이터가 없습니다."  # 에러 메시지와 함께 None 반환

    if len(X_train) == 0:  # 학습 데이터가 비어있는 경우
        return None, None, None, "학습용 데이터가 부족합니다."  # 에러 메시지와 함께 None 반환

    model = copy.deepcopy(best_model)  # 최적 모델을 deepcopy(깊은 복사)하여 독립적인 모델 생성

    # --- 내부 검증에서 찾은 최적 파라미터로 학습 (하이퍼파라미터 재튜닝 없음, 속도 우선) ---
    model.fit(X_train, y_train)  # 홀드아웃 지역 제외 데이터로 모델 학습
    preds = model.predict(X_test)  # 홀드아웃 지역 데이터에 대해 예측 수행

    rmse = np.sqrt(mean_squared_error(y_test, preds))  # RMSE(평균 제곱근 오차) 계산
    mae = mean_absolute_error(y_test, preds)  # MAE(평균 절대 오차) 계산
    r2 = r2_score(y_test, preds) if len(y_test) > 1 else np.nan  # R²(결정계수) 계산 (샘플 2개 이상인 경우만)

    return rmse, mae, r2, None  # RMSE, MAE, R²와 에러 메시지(None=성공)를 반환(리턴)


# ==========================================
# Premium Control Module(프리미엄 제어 모듈) Functions(함수) (V4.8)
# ==========================================

# --- 동적 가격 시뮬레이션 함수: 가격 탄력성 기반 수요 분산 ---
def simulate_dynamic_pricing(hourly_series, elasticity=-0.2, peak_surcharge=0.0, discount_rate=0.0, peak_hours=None, off_peak_hours=None):  # 동적 가격 시뮬레이션 함수 정의: 시간대별 부하와 가격 파라미터를 받음
    """
    가격 탄력성을 반영하여 시간대별 수요를 분산시키는 시뮬레이션 엔진.
    - hourly_series: pd.Series 또는 np.ndarray (24개 시간대의 충전 부하 데이터)
    - elasticity: 가격 탄력성 계수 (기본값 -0.2, 비탄력적)
    - peak_surcharge: peak 시간대 요금 할증률 (예: 0.20 = 20%)
    - discount_rate: off-peak 시간대 요금 할인율 (예: 0.15 = 15%)
    """
    if peak_hours is None:  # peak_hours(피크 시간대)가 지정되지 않은 경우
        peak_hours = [10, 11, 13, 14, 15, 16, 18, 19, 20, 21]  # 기본 피크 시간대 설정 (오전 10시~오후 9시 중 주요 시간)
    if off_peak_hours is None:  # off_peak_hours(비피크 시간대)가 지정되지 않은 경우
        off_peak_hours = [23, 0, 1, 2, 3, 4, 5, 6, 7, 8]  # 기본 비피크 시간대 설정 (심야~이른 아침)

    hourly_array = np.array(hourly_series, dtype=float)  # 시간대별 부하 데이터를 float(부동소수점) numpy array(배열)로 변환
    n_hours = len(hourly_array)  # 시간대 데이터의 요소 수 확인
    if n_hours != 24:  # 데이터가 24개 요소가 아닌 경우
        raise ValueError("시간대별 부하 데이터는 24개 요소로 이루어져야 합니다.")  # ValueError(값 오류) 예외 발생

    original_total = np.sum(hourly_array)  # 원본 총 수요량 계산 (수요 보존을 위해 저장)
    if original_total == 0:  # 총 수요가 0인 경우
        return hourly_array, np.zeros(24)  # 원본 배열과 제로 가격 변화를 반환

    # --- 시간대별 가격 변화율 벡터(price change vector) 계산 ---
    price_change = np.zeros(24)  # 24시간 가격 변화율 배열을 0으로 초기화
    for h in range(24):  # 24시간을 순회
        if h in peak_hours:  # 피크 시간대인 경우
            price_change[h] = peak_surcharge  # 할증률(surcharge)을 적용
        elif h in off_peak_hours:  # 비피크 시간대인 경우
            price_change[h] = -discount_rate  # 할인율(discount)을 음수로 적용

    # --- 가격 탄력성(elasticity)에 기반한 수요 변화량 계산 ---
    delta_demand = hourly_array * price_change * elasticity  # 수요 변화량 = 현재 수요 × 가격 변화율 × 탄력성 계수

    # --- 피크 시간대에서 감소된 수요 계산 (가격 인상으로 인한 수요 감소) ---
    reductions = np.minimum(delta_demand, 0.0)  # 음수 변화량(수요 감소분)만 추출
    total_reduced_demand = -np.sum(reductions)  # 총 감소 수요량 계산 (양수로 변환)

    # --- 시뮬레이션된 수요 초기 계산 ---
    simulated_demand = hourly_array + delta_demand  # 원본 수요에 변화량을 적용

    # --- 감소된 피크 수요를 비피크 시간대로 재분배 ---
    off_peak_mask = np.array([1 if h in off_peak_hours else 0 for h in range(24)])  # 비피크 시간대 mask(마스크) 배열 생성
    off_peak_sum = np.sum(hourly_array * off_peak_mask)  # 비피크 시간대의 총 원본 수요 합계

    if off_peak_sum > 0:  # 비피크 시간대에 수요가 존재하는 경우
        simulated_demand += total_reduced_demand * (hourly_array * off_peak_mask) / off_peak_sum  # 비피크 시간대 수요 비율에 따라 감소분을 재분배

    # --- 음수 수요 방지 ---
    simulated_demand = np.maximum(simulated_demand, 0.0)  # 수요가 음수가 되지 않도록 0으로 clip(클리핑)

    # --- 총 수요 보존을 위한 재정규화(re-normalization) ---
    new_total = np.sum(simulated_demand)  # 시뮬레이션 후 총 수요 계산
    if new_total > 0:  # 총 수요가 양수인 경우
        simulated_demand = simulated_demand * (original_total / new_total)  # 원본 총 수요와 일치하도록 비율 조정

    return simulated_demand, price_change  # 시뮬레이션된 수요 배열과 가격 변화율 배열을 반환(리턴)


# --- TOPSIS(이상적 해 유사도 기반 순위 기법) 다중 기준 의사결정 함수 ---
def calculate_topsis_rankings(data, weights=None):  # TOPSIS 순위 계산 함수 정의: 데이터와 가중치를 받음
    """
    TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
    다중 기준 의사결정(MCDA) 알고리즘을 사용하여 최적 충전소 입지 우선순위를 도출합니다.
    """
    if data.empty:  # 입력 데이터가 비어있는 경우
        return data  # 빈 데이터를 그대로 반환

    df = data.copy()  # 원본 보호를 위해 데이터 복사

    # --- 기본 가중치 설정 (사용자 지정이 없는 경우) ---
    if weights is None:  # 가중치가 지정되지 않은 경우
        weights = {  # 기본 가중치 딕셔너리(dictionary) 정의
            "전력_부하지수": 0.35,  # 전력 부하 지수 가중치: 35%
            "인프라_부하지수": 0.35,  # 인프라 부하 지수 가중치: 35%
            "충전소_밀집도_역수": 0.15,  # 충전소 밀집도 역수 가중치: 15%
            "전력망_완화율": 0.15  # 전력망 완화율 가중치: 15%
        }

    # --- 1단계: 평가 feature(특성) 행렬 계산 ---
    # 충전소 밀집도 역수 = 1 / (충전소 개수 + 1): 충전소가 적을수록 값이 커짐
    if "충전소개수" in df.columns:  # 충전소개수 column(컬럼)이 존재하는 경우
        df["충전소_밀집도_역수"] = 1.0 / (df["충전소개수"] + 1.0)  # 충전소 개수의 역수를 밀집도 역수로 계산
    else:  # 충전소개수 column(컬럼)이 없는 경우
        df["충전소_밀집도_역수"] = 1.0 / (df["전체_충전기대수"] * 0.1 + 1.0)  # 전체 충전기 대수를 활용한 대안 계산

    # --- 설치 비용 대비 전력망 완화율 계산 ---
    if "총용량_kW" in df.columns and "총_전력판매량" in df.columns:  # 필요한 column(컬럼)들이 모두 존재하는 경우
        df["전력망_완화율"] = df["총용량_kW"] / (df["총_전력판매량"] + 1.0)  # 총 용량 / 총 전력판매량으로 완화율 계산
    else:  # 필요한 column(컬럼)이 없는 경우
        df["전력망_완화율"] = 1.0 / (df["전력_부하지수"] + 1.0)  # 부하 지수의 역수를 대안으로 사용

    features = list(weights.keys())  # 평가에 사용할 feature(특성) 이름 목록 추출

    # --- 2단계: Decision Matrix(의사결정 행렬) 생성 ---
    X = df[features].values.astype(float)  # 선택된 feature(특성)들을 float(부동소수점) numpy array(배열)로 변환

    # --- 3단계: Vector Normalization(벡터 정규화) ---
    norm = np.sqrt(np.sum(X**2, axis=0))  # 각 feature(특성) column(컬럼)의 L2 norm(노름) 계산
    norm = np.where(norm == 0, 1e-5, norm)  # norm(노름)이 0인 경우 epsilon(엡실론) 값으로 대체하여 0 나누기 방지
    R = X / norm  # 정규화된 의사결정 행렬(R) 생성

    # --- 4단계: Weighted Normalized Decision Matrix(가중치 정규화 의사결정 행렬) 계산 ---
    w = np.array([weights[f] for f in features])  # 가중치를 numpy array(배열)로 변환
    w = w / np.sum(w)  # 가중치 합이 1이 되도록 정규화
    V = R * w  # 정규화된 행렬에 가중치를 적용

    # --- 5단계: PIS(양의 이상적 해)와 NIS(음의 이상적 해) 결정 ---
    pis = np.max(V, axis=0)  # PIS(Positive Ideal Solution, 양의 이상적 해): 각 feature(특성)의 최대값
    nis = np.min(V, axis=0)  # NIS(Negative Ideal Solution, 음의 이상적 해): 각 feature(특성)의 최소값

    # --- 6단계: Separation Measures(분리 척도) 계산 ---
    S_plus = np.sqrt(np.sum((V - pis)**2, axis=1))  # S+(양의 이상적 해까지의 유클리드 거리) 계산
    S_minus = np.sqrt(np.sum((V - nis)**2, axis=1))  # S-(음의 이상적 해까지의 유클리드 거리) 계산

    # --- 7단계: Relative Closeness Score(상대적 근접도, C_i) 계산 ---
    denom = S_plus + S_minus  # 분모: S+와 S-의 합
    denom = np.where(denom == 0, 1e-5, denom)  # 분모가 0인 경우 epsilon(엡실론)으로 대체하여 0 나누기 방지
    closeness = S_minus / denom  # 상대적 근접도 계산: S- / (S+ + S-), 1에 가까울수록 이상적

    df["TOPSIS_점수"] = closeness  # TOPSIS 점수 column(컬럼)에 상대적 근접도 저장
    df["TOPSIS_순위"] = df["TOPSIS_점수"].rank(ascending=False, method="min").astype(int)  # TOPSIS 점수 기준 내림차순 순위 계산

    return df  # TOPSIS 점수와 순위가 추가된 DataFrame(데이터프레임) 반환(리턴)
