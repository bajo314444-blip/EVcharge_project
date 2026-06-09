# ============================================================
# 파일명: models.py
# 설명: 전기차 충전 인프라 부하 예측을 위한 머신러닝/딥러닝 모델 정의,
#       학습, 평가 파이프라인을 포함하는 모듈
# ============================================================

import numpy as np  # numpy(넘파이) 수치 연산 라이브러리를 np로 import(임포트)
import pandas as pd  # pandas(판다스) 데이터 분석 라이브러리를 pd로 import(임포트)
import streamlit as st  # streamlit(스트림릿) 웹 앱 프레임워크를 st로 import(임포트)
from io import StringIO  # StringIO(문자열 입출력) 클래스를 import(임포트)
from sklearn.ensemble import (  # sklearn(사이킷런) ensemble(앙상블) 모듈에서 회귀 모델들을 import(임포트)
    ExtraTreesRegressor,  # ExtraTreesRegressor(엑스트라 트리 회귀기) 클래스 import(임포트)
    GradientBoostingRegressor,  # GradientBoostingRegressor(그래디언트 부스팅 회귀기) 클래스 import(임포트)
    RandomForestRegressor,  # RandomForestRegressor(랜덤 포레스트 회귀기) 클래스 import(임포트)
)
from sklearn.inspection import permutation_importance  # permutation_importance(순열 중요도) 함수를 import(임포트)
from sklearn.linear_model import Ridge  # Ridge(릿지) 회귀 모델 클래스를 import(임포트)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # 평가 지표(MAE, MSE, R2) 함수들을 import(임포트)
from sklearn.model_selection import train_test_split, GridSearchCV, KFold  # 데이터 분할, GridSearch(그리드서치), KFold(K-폴드) 교차검증 import(임포트)
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors  # KNN(K-최근접 이웃) 회귀기 및 NearestNeighbors(최근접 이웃) 클래스 import(임포트)
from sklearn.pipeline import Pipeline  # Pipeline(파이프라인) 클래스를 import(임포트)
from sklearn.preprocessing import MinMaxScaler  # MinMaxScaler(최소-최대 스케일러) 전처리 클래스를 import(임포트)
import warnings  # warnings(경고) 모듈을 import(임포트)

# --- CustomSMOTE(커스텀 SMOTE) 클래스: 소수 클래스 오버샘플링 구현 ---
class CustomSMOTE:  # CustomSMOTE(커스텀 합성 소수 오버샘플링) 클래스 정의
    def __init__(self, k_neighbors=5, random_state=42):  # 생성자: k_neighbors(이웃 수)와 random_state(난수 시드)를 초기화
        self.k_neighbors = k_neighbors  # k_neighbors(이웃 수) 인스턴스 변수에 저장
        self.random_state = random_state  # random_state(난수 시드) 인스턴스 변수에 저장

    # --- 오버샘플링 수행 메서드 ---
    def fit_resample(self, X, y):  # fit_resample(학습 및 리샘플링) 메서드 정의: 입력 X, 라벨 y를 받음
        np.random.seed(self.random_state)  # 재현성을 위해 난수 seed(시드) 설정
        X_arr = np.array(X)  # 입력 X를 numpy array(배열)로 변환
        y_arr = np.array(y)  # 라벨 y를 numpy array(배열)로 변환

        unique_classes, class_counts = np.unique(y_arr, return_counts=True)  # 고유 클래스와 각 클래스별 개수를 계산
        max_count = class_counts.max()  # 가장 많은 클래스의 샘플 수(최대 개수)를 구함

        X_resampled = [X_arr]  # 리샘플링된 X 리스트에 원본 데이터를 초기값으로 추가
        y_resampled = [y_arr]  # 리샘플링된 y 리스트에 원본 라벨을 초기값으로 추가

        # --- 각 클래스별 오버샘플링 반복 ---
        for cls, count in zip(unique_classes, class_counts):  # 각 클래스(cls)와 해당 개수(count)를 순회
            if count == max_count:  # 이미 최대 개수인 클래스는 건너뜀
                continue  # 다음 클래스로 이동

            cls_indices = np.where(y_arr == cls)[0]  # 현재 클래스에 해당하는 인덱스(index) 배열 추출
            cls_X = X_arr[cls_indices]  # 현재 클래스의 feature(특성) 데이터만 추출

            n_samples_to_create = max_count - count  # 생성해야 할 합성 샘플 수 계산
            if n_samples_to_create <= 0:  # 생성할 샘플이 없으면 건너뜀
                continue  # 다음 클래스로 이동

            k = min(self.k_neighbors, count - 1)  # k_neighbors(이웃 수)와 (클래스 샘플 수 - 1) 중 작은 값 선택
            if k < 1:  # k가 1 미만이면 이웃 기반 합성이 불가능하므로 단순 복제
                indices = np.random.choice(cls_indices, size=n_samples_to_create, replace=True)  # 랜덤으로 인덱스를 복원 추출
                X_resampled.append(X_arr[indices])  # 복제된 feature(특성) 데이터를 리스트에 추가
                y_resampled.append(np.full(n_samples_to_create, cls))  # 해당 클래스 라벨을 채워서 추가
                continue  # 다음 클래스로 이동

            nn = NearestNeighbors(n_neighbors=k + 1)  # NearestNeighbors(최근접 이웃) 객체 생성 (자기 자신 포함이므로 k+1)
            nn.fit(cls_X)  # 현재 클래스 데이터로 NearestNeighbors(최근접 이웃) 모델 학습
            neighbors = nn.kneighbors(cls_X, return_distance=False)[:, 1:]  # 각 샘플의 k개 이웃 인덱스 추출 (자기 자신 제외)

            synthetic_X = []  # 합성된 feature(특성) 데이터를 저장할 리스트 초기화
            # --- 합성 샘플 생성 반복 ---
            for _ in range(n_samples_to_create):  # 필요한 합성 샘플 수만큼 반복
                idx = np.random.randint(0, count)  # 현재 클래스에서 랜덤으로 기준 샘플 인덱스 선택
                neighbor_idx = np.random.choice(neighbors[idx])  # 선택된 기준 샘플의 이웃 중 하나를 랜덤 선택

                diff = cls_X[neighbor_idx] - cls_X[idx]  # 기준 샘플과 이웃 샘플 간의 차이 벡터(diff) 계산
                gap = np.random.rand()  # 0~1 사이의 랜덤 보간(interpolation) 비율 생성
                synthetic_sample = cls_X[idx] + gap * diff  # 보간(interpolation)을 통해 합성 샘플 생성
                synthetic_X.append(synthetic_sample)  # 합성된 샘플을 리스트에 추가

            X_resampled.append(np.array(synthetic_X))  # 합성된 feature(특성) 배열을 리샘플링 리스트에 추가
            y_resampled.append(np.full(n_samples_to_create, cls))  # 해당 클래스 라벨 배열을 리샘플링 리스트에 추가

        X_out = np.vstack(X_resampled)  # 모든 리샘플링된 X 데이터를 수직으로 결합(vstack)
        y_out = np.concatenate(y_resampled)  # 모든 리샘플링된 y 라벨을 연결(concatenate)

        if isinstance(X, pd.DataFrame):  # 원본 X가 DataFrame(데이터프레임)인 경우
            X_out = pd.DataFrame(X_out, columns=X.columns)  # 결과를 동일한 column(컬럼)명의 DataFrame(데이터프레임)으로 변환

        return X_out, y_out  # 리샘플링된 X와 y를 반환(리턴)

# --- SMAPE(대칭 평균 절대 백분율 오차) 계산 함수 ---
def smape(y_true, y_pred):  # SMAPE(대칭 평균 절대 백분율 오차) 계산 함수 정의
    y_true = np.array(y_true)  # 실제값을 numpy array(배열)로 변환
    y_pred = np.array(y_pred)  # 예측값을 numpy array(배열)로 변환
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100  # SMAPE 공식 적용 후 백분율로 반환

# --- Softmax(소프트맥스) 활성화 함수 ---
def softmax(x, axis=-1):  # softmax(소프트맥스) 함수 정의: 입력 x와 축(axis)을 받음
    x = x - np.max(x, axis=axis, keepdims=True)  # 수치 안정성을 위해 최대값을 빼서 overflow(오버플로우) 방지
    exp_x = np.exp(x)  # 지수(exponential) 함수 적용
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)  # 합으로 나누어 확률 분포로 정규화하여 반환

# --- NumpyCNN1DRegressor: Numpy 기반 1D CNN(합성곱 신경망) 회귀 모델 클래스 ---
class NumpyCNN1DRegressor:  # 1D CNN(1차원 합성곱 신경망) 회귀 모델 클래스 정의
    """Small 1D-CNN regressor for tabular feature vectors."""

    def __init__(self, n_filters=10, kernel_size=3, epochs=700, lr=0.015, random_state=42):  # 생성자: 필터 수, 커널 크기, 에폭, 학습률, 난수 시드 초기화
        self.n_filters = n_filters  # n_filters(필터 수) 인스턴스 변수에 저장
        self.kernel_size = kernel_size  # kernel_size(커널 크기) 인스턴스 변수에 저장
        self.epochs = epochs  # epochs(학습 에폭 수) 인스턴스 변수에 저장
        self.lr = lr  # lr(learning rate, 학습률) 인스턴스 변수에 저장
        self.random_state = random_state  # random_state(난수 시드) 인스턴스 변수에 저장

    # --- 모델 학습 메서드 ---
    def fit(self, X, y):  # fit(학습) 메서드 정의: 입력 X와 타겟 y를 받음
        X = np.asarray(X, dtype=float)  # X를 float(부동소수점) 타입 numpy array(배열)로 변환
        y = np.asarray(y, dtype=float)  # y를 float(부동소수점) 타입 numpy array(배열)로 변환
        self.x_mean_ = X.mean(axis=0)  # X의 feature(특성)별 평균값 저장 (정규화용)
        self.x_std_ = X.std(axis=0) + 1e-8  # X의 feature(특성)별 표준편차 저장 (0 나누기 방지를 위해 epsilon(엡실론) 추가)
        self.y_mean_ = y.mean()  # y의 평균값 저장 (타겟 정규화용)
        self.y_std_ = y.std() + 1e-8  # y의 표준편차 저장 (0 나누기 방지를 위해 epsilon(엡실론) 추가)
        Xs = (X - self.x_mean_) / self.x_std_  # X를 표준 정규화(z-score normalization) 수행
        ys = (y - self.y_mean_) / self.y_std_  # y를 표준 정규화(z-score normalization) 수행

        rng = np.random.default_rng(self.random_state)  # 난수 생성기(rng) 초기화
        n_features = Xs.shape[1]  # 입력 feature(특성) 개수 저장
        self.kernel_size_ = min(self.kernel_size, n_features)  # 커널 크기가 feature(특성) 수를 초과하지 않도록 조정
        n_pos = n_features - self.kernel_size_ + 1  # convolution(합성곱) 적용 가능한 위치(position) 수 계산
        self.w_conv_ = rng.normal(0, 0.12, size=(self.n_filters, self.kernel_size_))  # convolution(합성곱) 가중치를 정규분포로 초기화
        self.b_conv_ = np.zeros(self.n_filters)  # convolution(합성곱) bias(편향)를 0으로 초기화
        self.w_out_ = rng.normal(0, 0.12, size=n_pos * self.n_filters)  # 출력층 가중치를 정규분포로 초기화
        self.b_out_ = 0.0  # 출력층 bias(편향)를 0으로 초기화

        # --- 학습 루프: epochs(에폭) 수만큼 반복 ---
        for _ in range(self.epochs):  # 설정된 에폭 수만큼 학습 반복
            z = self._conv_forward(Xs)  # forward pass(순전파): convolution(합성곱) 연산 수행
            h = np.maximum(z, 0)  # ReLU(렐루) 활성화 함수 적용 (음수를 0으로 변환)
            flat = h.reshape(len(Xs), -1)  # 3D 텐서를 2D로 평탄화(flatten)
            pred = flat @ self.w_out_ + self.b_out_  # 출력층 선형 연산으로 예측값 계산
            grad_pred = 2 * (pred - ys) / len(Xs)  # MSE loss(손실)의 예측값에 대한 gradient(기울기) 계산

            # --- 역전파(backpropagation) 계산 ---
            grad_w_out = flat.T @ grad_pred  # 출력층 가중치에 대한 gradient(기울기) 계산
            grad_b_out = grad_pred.sum()  # 출력층 bias(편향)에 대한 gradient(기울기) 계산
            grad_flat = np.outer(grad_pred, self.w_out_)  # 평탄화 레이어에 대한 gradient(기울기) 계산
            grad_h = grad_flat.reshape(h.shape)  # gradient(기울기)를 원래 shape(형상)으로 복원
            grad_z = grad_h * (z > 0)  # ReLU(렐루)의 gradient(기울기) 적용 (양수인 부분만 전파)

            grad_w_conv = np.zeros_like(self.w_conv_)  # convolution(합성곱) 가중치 gradient(기울기) 초기화
            grad_b_conv = grad_z.sum(axis=(0, 1))  # convolution(합성곱) bias(편향) gradient(기울기) 계산
            for pos in range(n_pos):  # 각 convolution(합성곱) 위치에 대해 반복
                window = Xs[:, pos : pos + self.kernel_size_]  # 현재 위치의 입력 window(윈도우) 추출
                grad_w_conv += grad_z[:, pos, :].T @ window  # convolution(합성곱) 가중치 gradient(기울기) 누적

            # --- 가중치 업데이트 (경사 하강법) ---
            self.w_out_ -= self.lr * grad_w_out  # 출력층 가중치를 learning rate(학습률)만큼 업데이트
            self.b_out_ -= self.lr * grad_b_out  # 출력층 bias(편향)를 learning rate(학습률)만큼 업데이트
            self.w_conv_ -= self.lr * grad_w_conv  # convolution(합성곱) 가중치를 learning rate(학습률)만큼 업데이트
            self.b_conv_ -= self.lr * grad_b_conv  # convolution(합성곱) bias(편향)를 learning rate(학습률)만큼 업데이트
        return self  # 학습 완료된 모델 자신을 반환(리턴)

    # --- 순전파(forward pass) 내부 메서드: Convolution(합성곱) 연산 ---
    def _conv_forward(self, Xs):  # convolution(합성곱) 순전파 내부 메서드 정의
        n_pos = Xs.shape[1] - self.kernel_size_ + 1  # convolution(합성곱) 적용 가능 위치 수 계산
        out = np.zeros((len(Xs), n_pos, self.n_filters))  # 출력 텐서를 0으로 초기화
        for pos in range(n_pos):  # 각 위치에 대해 convolution(합성곱) 수행
            window = Xs[:, pos : pos + self.kernel_size_]  # 현재 위치의 입력 window(윈도우) 추출
            out[:, pos, :] = window @ self.w_conv_.T + self.b_conv_  # 행렬 곱과 bias(편향) 덧셈으로 convolution(합성곱) 계산
        return out  # convolution(합성곱) 출력 텐서 반환(리턴)

    # --- 예측 메서드 ---
    def predict(self, X):  # predict(예측) 메서드 정의: 새로운 입력 X에 대해 예측 수행
        X = np.asarray(X, dtype=float)  # X를 float(부동소수점) numpy array(배열)로 변환
        Xs = (X - self.x_mean_) / self.x_std_  # 학습 시 저장된 통계로 입력 데이터 정규화
        h = np.maximum(self._conv_forward(Xs), 0).reshape(len(Xs), -1)  # convolution(합성곱) → ReLU(렐루) → 평탄화(flatten) 수행
        pred = h @ self.w_out_ + self.b_out_  # 출력층 선형 연산으로 정규화된 예측값 계산
        return pred * self.y_std_ + self.y_mean_  # 역정규화(denormalization)하여 원래 스케일로 복원 후 반환

    # --- R² 평가 메서드 ---
    def score(self, X, y):  # score(평가) 메서드 정의: R²(결정계수)를 반환
        return r2_score(y, self.predict(X))  # 실제값과 예측값의 R²(결정계수) 계산 후 반환

# --- TabularTransformerRegressor: Transformer(트랜스포머) 기반 테이블 데이터 회귀 모델 ---
class TabularTransformerRegressor:  # TabularTransformerRegressor(테이블 트랜스포머 회귀기) 클래스 정의
    """Lightweight transformer-style feature encoder plus ridge regression head."""

    def __init__(self, d_model=8, random_state=42, ridge_alpha=1.0):  # 생성자: d_model(모델 차원), random_state(난수 시드), ridge_alpha(릿지 정규화 계수) 초기화
        self.d_model = d_model  # d_model(모델 임베딩 차원 수) 인스턴스 변수에 저장
        self.random_state = random_state  # random_state(난수 시드) 인스턴스 변수에 저장
        self.ridge_alpha = ridge_alpha  # ridge_alpha(릿지 정규화 강도) 인스턴스 변수에 저장

    # --- 모델 학습 메서드 ---
    def fit(self, X, y):  # fit(학습) 메서드 정의: 입력 X와 타겟 y로 모델 학습
        X = np.asarray(X, dtype=float)  # X를 float(부동소수점) numpy array(배열)로 변환
        y = np.asarray(y, dtype=float)  # y를 float(부동소수점) numpy array(배열)로 변환
        self.x_mean_ = X.mean(axis=0)  # X의 feature(특성)별 평균값 저장
        self.x_std_ = X.std(axis=0) + 1e-8  # X의 feature(특성)별 표준편차 저장 (0 나누기 방지)
        rng = np.random.default_rng(self.random_state)  # 난수 생성기(rng) 초기화
        self.n_features_ = X.shape[1]  # feature(특성) 개수를 인스턴스 변수에 저장
        self.token_weight_ = rng.normal(0, 0.35, size=(self.n_features_, self.d_model))  # token(토큰) 투영 가중치 행렬을 정규분포로 초기화
        self.feature_embed_ = rng.normal(0, 0.15, size=(self.n_features_, self.d_model))  # feature(특성) embedding(임베딩) 행렬을 정규분포로 초기화
        self.w_q_ = rng.normal(0, 0.25, size=(self.d_model, self.d_model))  # Query(쿼리) 가중치 행렬 초기화
        self.w_k_ = rng.normal(0, 0.25, size=(self.d_model, self.d_model))  # Key(키) 가중치 행렬 초기화
        self.w_v_ = rng.normal(0, 0.25, size=(self.d_model, self.d_model))  # Value(값) 가중치 행렬 초기화
        z = self._encode((X - self.x_mean_) / self.x_std_)  # 정규화된 X를 Transformer(트랜스포머) 인코더(encoder)로 인코딩
        self.head_ = Ridge(alpha=self.ridge_alpha)  # Ridge(릿지) 회귀 헤드(head) 생성
        self.head_.fit(z, y)  # 인코딩된 feature(특성)로 Ridge(릿지) 회귀 학습
        return self  # 학습 완료된 모델 자신을 반환(리턴)

    # --- Transformer(트랜스포머) 인코딩 내부 메서드 ---
    def _encode(self, Xs):  # _encode(인코딩) 내부 메서드: 입력을 Transformer(트랜스포머) 구조로 인코딩
        tokens = Xs[:, :, None] * self.token_weight_[None, :, :] + self.feature_embed_[None, :, :]  # 각 feature(특성)를 token(토큰)으로 변환 후 positional embedding(위치 임베딩) 추가
        q = tokens @ self.w_q_  # Query(쿼리) 행렬 계산
        k = tokens @ self.w_k_  # Key(키) 행렬 계산
        v = tokens @ self.w_v_  # Value(값) 행렬 계산
        attn = softmax((q @ np.swapaxes(k, 1, 2)) / np.sqrt(self.d_model), axis=-1)  # Scaled Dot-Product Attention(스케일드 닷 프로덕트 어텐션) 가중치 계산
        encoded = attn @ v  # Attention(어텐션) 가중치를 Value(값)에 적용하여 인코딩 결과 생성
        pooled = encoded.mean(axis=1)  # 시퀀스(sequence) 축을 평균 풀링(mean pooling)
        flat = encoded.reshape(len(Xs), -1)  # 인코딩 결과를 평탄화(flatten)
        return np.hstack([Xs, pooled, flat])  # 원본 입력, 풀링 결과, 평탄화 결과를 수평 결합(hstack)하여 반환

    # --- 예측 메서드 ---
    def predict(self, X):  # predict(예측) 메서드: 새로운 입력 X에 대해 예측 수행
        X = np.asarray(X, dtype=float)  # X를 float(부동소수점) numpy array(배열)로 변환
        z = self._encode((X - self.x_mean_) / self.x_std_)  # 정규화된 X를 Transformer(트랜스포머) 인코더로 인코딩
        return self.head_.predict(z)  # Ridge(릿지) 회귀 헤드를 통해 최종 예측값 반환

    # --- R² 평가 메서드 ---
    def score(self, X, y):  # score(평가) 메서드: R²(결정계수)를 반환
        return r2_score(y, self.predict(X))  # 실제값과 예측값의 R²(결정계수) 계산 후 반환

# --- Feature Matrix(특성 행렬) 생성 함수 ---
def make_feature_matrix(final):  # make_feature_matrix(특성 행렬 생성) 함수 정의: 최종 데이터를 받아 ML 입력 행렬 생성
    ml = final.copy()  # 원본 데이터를 복사하여 변경 방지
    ml = pd.get_dummies(ml, columns=["시도", "용도"], dtype=float)  # 시도와 용도 column(컬럼)을 one-hot encoding(원-핫 인코딩)으로 변환
    feature_cols = ["전기차_전체대수", "충전인프라_규모_PCA", "충전기_1대당_평균용량", "인프라_부하지수"]  # 기본 수치형 feature(특성) column(컬럼) 목록 정의
    feature_cols += [c for c in ml.columns if c.startswith("시도_") or c.startswith("용도_")]  # one-hot encoding(원-핫 인코딩)된 시도/용도 column(컬럼)을 feature(특성) 목록에 추가
    X = ml[feature_cols].rename(  # 선택된 feature(특성) column(컬럼)들을 영문 이름으로 rename(이름 변경)
        columns={
            "전기차_전체대수": "total_ev_count",  # 전기차 전체 대수 → total_ev_count로 rename(이름 변경)
            "충전인프라_규모_PCA": "infra_size_pca",  # 충전인프라 규모 PCA → infra_size_pca로 rename(이름 변경)
            "충전기_1대당_평균용량": "avg_capacity_per_charger",  # 충전기 1대당 평균 용량 → avg_capacity_per_charger로 rename(이름 변경)
            "인프라_부하지수": "infra_load_index",  # 인프라 부하지수 → infra_load_index로 rename(이름 변경)
            "시도_경기": "region_gyeonggi",  # 시도_경기 → region_gyeonggi로 rename(이름 변경)
            "시도_서울": "region_seoul",  # 시도_서울 → region_seoul로 rename(이름 변경)
            "시도_인천": "region_incheon",  # 시도_인천 → region_incheon로 rename(이름 변경)
            "용도_사업자용": "usage_business",  # 용도_사업자용 → usage_business로 rename(이름 변경)
            "용도_자가용": "usage_private",  # 용도_자가용 → usage_private로 rename(이름 변경)
        }
    )

    # --- 기본 Time Spike(시간 스파이크) column(컬럼) 추가 (0으로 초기화) ---
    for col in ["is_commute_time", "is_holiday", "is_golden_week", "is_weekend"]:  # 시간 시나리오 관련 column(컬럼)들을 순회
        if col not in X.columns:  # 해당 column(컬럼)이 없으면 추가
            X[col] = 0.0  # 해당 column(컬럼)을 0.0으로 초기화

    # --- 고정된 feature(특성) 목록 및 순서 강제 ---
    FIXED_FEATURES = [  # 모델에 사용되는 고정 feature(특성) 목록 정의
        "total_ev_count",  # 전기차 전체 대수
        "infra_size_pca",  # 충전인프라 규모 PCA 점수
        "avg_capacity_per_charger",  # 충전기 1대당 평균 용량
        "infra_load_index",  # 인프라 부하 지수
        "region_gyeonggi",  # 경기 지역 여부 (one-hot)
        "region_seoul",  # 서울 지역 여부 (one-hot)
        "region_incheon",  # 인천 지역 여부 (one-hot)
        "usage_business",  # 사업자용 여부 (one-hot)
        "usage_private",  # 자가용 여부 (one-hot)
        "is_commute_time",  # 출퇴근 시간대 여부
        "is_holiday",  # 공휴일 여부
        "is_golden_week",  # 연휴(골든위크) 여부
        "is_weekend"  # 주말 여부
    ]
    X = X.reindex(columns=FIXED_FEATURES, fill_value=0.0)  # 고정된 feature(특성) 순서로 재정렬하고 없는 column(컬럼)은 0으로 채움
    return X  # 완성된 feature matrix(특성 행렬)를 반환(리턴)

# --- 회귀용 SMOTE 오버샘플링 적용 함수 ---
def apply_smote_for_regression(X_train, y_train, n_bins=5):  # 회귀 타겟을 구간(bin) 분할 후 SMOTE 적용하는 함수 정의
    bins = np.quantile(y_train, np.linspace(0, 1, n_bins + 1))  # y_train을 n_bins(구간 수)개의 분위수로 분할하여 경계값 계산
    bins[0] = -np.inf  # 첫 번째 경계를 음의 무한대(-∞)로 설정
    bins[-1] = np.inf  # 마지막 경계를 양의 무한대(+∞)로 설정
    y_binned = np.digitize(y_train, bins)  # 연속 타겟값을 구간(bin) 인덱스로 이산화(discretize)

    class_counts = pd.Series(y_binned).value_counts()  # 각 구간(bin)별 샘플 수를 계산
    min_count = class_counts.min()  # 가장 적은 구간(bin)의 샘플 수 확인
    if min_count < 2:  # 최소 샘플 수가 2 미만이면 SMOTE 적용 불가
        return X_train, y_train  # 원본 데이터를 그대로 반환

    k_neighbors = min(5, min_count - 1)  # k_neighbors(이웃 수)를 최소 샘플 수에 맞게 조정
    smote = CustomSMOTE(k_neighbors=k_neighbors, random_state=42)  # CustomSMOTE(커스텀 SMOTE) 객체 생성

    try:  # SMOTE 적용 시도
        X_resampled, _ = smote.fit_resample(X_train, y_binned)  # 구간(bin) 라벨 기준으로 오버샘플링 수행
        knn = KNeighborsRegressor(n_neighbors=k_neighbors)  # KNN(K-최근접 이웃) 회귀기를 생성하여 연속 타겟값 복원에 사용
        knn.fit(X_train, y_train)  # 원본 데이터로 KNN(K-최근접 이웃) 회귀기 학습
        y_resampled = knn.predict(X_resampled)  # 오버샘플링된 X에 대해 연속 타겟값 예측
        if isinstance(y_train, pd.Series):  # y_train이 Series(시리즈)인 경우
            y_resampled = pd.Series(y_resampled)  # 예측 결과도 Series(시리즈)로 변환
        y_resampled[:len(y_train)] = y_train  # 원본 데이터 부분은 실제 타겟값으로 복원 (정확성 보장)
        return X_resampled, y_resampled  # 오버샘플링된 X와 y를 반환(리턴)
    except ValueError:  # SMOTE 적용 중 ValueError(값 오류) 발생 시
        return X_train, y_train  # 원본 데이터를 그대로 반환

# --- 전체 모델 학습 및 평가 파이프라인 함수 (Streamlit 캐시 적용) ---
@st.cache_data(show_spinner="예측 모델을 학습하고 평가하는 중입니다...")  # Streamlit cache(캐시) 데코레이터: 동일 입력 시 재계산 방지
def train_models(final_json, use_smote=False):  # train_models(모델 학습) 함수 정의: JSON 데이터와 SMOTE 사용 여부를 받음
    final = pd.read_json(StringIO(final_json), orient="split")  # JSON 문자열을 DataFrame(데이터프레임)으로 파싱(parsing)
    X = make_feature_matrix(final)  # feature matrix(특성 행렬) 생성 함수 호출
    y = final["전력_부하지수"].astype(float)  # 타겟 변수(전력 부하지수)를 float(부동소수점)으로 변환
    row_info = final[["지역", "용도", "전력_부하지수"]].copy()  # 결과 분석용 메타 정보(지역, 용도, 부하지수) 복사

    # --- 데이터 분할: Train(학습) 70%, Validation(검증) 20%, Test(테스트) 10% ---
    X_train, X_temp, y_train, y_temp, info_train, info_temp = train_test_split(  # 전체 데이터를 학습용 70%와 나머지 30%로 분할
        X, y, row_info, test_size=0.3, random_state=42  # test_size=0.3으로 30% 분리, 재현성을 위해 random_state(난수 시드)=42 설정
    )
    X_val, X_test, y_val, y_test, info_val, info_test = train_test_split(  # 나머지 30%를 검증용과 테스트용으로 분할
        X_temp, y_temp, info_temp, test_size=1 / 3, random_state=42  # test_size=1/3으로 테스트 10% 분리
    )

    if use_smote:  # SMOTE 사용이 활성화된 경우
        X_train, y_train = apply_smote_for_regression(X_train, y_train)  # 학습 데이터에 회귀용 SMOTE 오버샘플링 적용

    # --- Time Spike(시간 스파이크) Data Augmentation(데이터 증강): 가상 시나리오 가중치 학습 ---
    X_commute = X_train.copy()  # 출퇴근 시나리오용 데이터 복사
    X_commute["is_commute_time"] = 1.0  # 출퇴근 시간대 flag(플래그)를 1로 설정
    y_commute = y_train * 1.3  # 출퇴근 시 부하 30% 증가 시나리오

    X_weekend = X_train.copy()  # 주말 시나리오용 데이터 복사
    X_weekend["is_weekend"] = 1.0  # 주말 flag(플래그)를 1로 설정
    y_weekend = y_train * 1.5  # 주말 시 부하 50% 증가 시나리오

    X_holiday = X_train.copy()  # 공휴일/연휴 시나리오용 데이터 복사
    X_holiday["is_holiday"] = 1.0  # 공휴일 flag(플래그)를 1로 설정
    X_holiday["is_golden_week"] = 1.0  # 연휴(골든위크) flag(플래그)를 1로 설정
    y_holiday = y_train * 2.0  # 연휴 시 부하 100% 증가 시나리오

    X_train = pd.concat([X_train, X_commute, X_weekend, X_holiday], ignore_index=True)  # 원본 + 증강 데이터를 수직 결합(concat)
    y_train = pd.concat([y_train, y_commute, y_weekend, y_holiday], ignore_index=True)  # 원본 + 증강 타겟값을 수직 결합(concat)

    # --- 데이터 셔플링(shuffling) ---
    idx = np.random.permutation(len(X_train))  # 랜덤 순열(permutation) 인덱스 생성
    X_train = X_train.iloc[idx].reset_index(drop=True)  # X_train을 랜덤 순서로 셔플링 후 인덱스 리셋
    y_train = y_train.iloc[idx].reset_index(drop=True)  # y_train을 동일한 순서로 셔플링 후 인덱스 리셋

    # --- RandomForest(랜덤 포레스트) 하이퍼파라미터 튜닝 ---
    rf_base = RandomForestRegressor(random_state=42)  # RandomForest(랜덤 포레스트) 기본 모델 생성
    rf_param_grid = {'max_depth': [3, 5, 10], 'min_samples_leaf': [1, 2, 4]}  # 탐색할 하이퍼파라미터 그리드 정의
    rf_grid = GridSearchCV(rf_base, rf_param_grid, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='neg_mean_squared_error')  # GridSearchCV(그리드서치 교차검증) 설정: 5-fold CV, MSE 기준
    rf_grid.fit(X_train, y_train)  # GridSearchCV(그리드서치)로 최적 하이퍼파라미터 탐색 및 학습
    best_rf = rf_grid.best_estimator_  # 최적 파라미터의 RandomForest(랜덤 포레스트) 모델 추출

    # --- GradientBoosting(그래디언트 부스팅) 하이퍼파라미터 튜닝 ---
    gb_base = GradientBoostingRegressor(random_state=42)  # GradientBoosting(그래디언트 부스팅) 기본 모델 생성
    gb_param_grid = {'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]}  # 탐색할 하이퍼파라미터 그리드 정의
    gb_grid = GridSearchCV(gb_base, gb_param_grid, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='neg_mean_squared_error')  # GridSearchCV(그리드서치 교차검증) 설정: 5-fold CV, MSE 기준
    gb_grid.fit(X_train, y_train)  # GridSearchCV(그리드서치)로 최적 하이퍼파라미터 탐색 및 학습
    best_gb = gb_grid.best_estimator_  # 최적 파라미터의 GradientBoosting(그래디언트 부스팅) 모델 추출

    # --- 모든 모델 딕셔너리(dictionary) 정의 ---
    models = {  # 모델 이름을 key(키)로, (그룹, 모델 객체) 튜플을 value(값)으로 하는 딕셔너리(dictionary) 정의
        "RandomForest (Tuned)": ("Machine Learning", best_rf),  # 튜닝된 RandomForest(랜덤 포레스트) 모델
        "ExtraTrees": ("Machine Learning", ExtraTreesRegressor(max_depth=5, min_samples_leaf=2, random_state=42)),  # ExtraTrees(엑스트라 트리) 모델 (고정 파라미터)
        "GradientBoosting (Tuned)": ("Machine Learning", best_gb),  # 튜닝된 GradientBoosting(그래디언트 부스팅) 모델
        "KNN": ("Machine Learning", Pipeline([("scale", MinMaxScaler()), ("model", KNeighborsRegressor(n_neighbors=5))])),  # KNN(K-최근접 이웃) 모델 (MinMaxScaler 파이프라인 포함)
        "Numpy_1D_CNN": ("Deep Learning - CNN", NumpyCNN1DRegressor(random_state=42)),  # Numpy 기반 1D CNN(합성곱 신경망) 모델
        "Tabular_Transformer": ("Deep Learning - Transformer", TabularTransformerRegressor(random_state=42)),  # Tabular Transformer(테이블 트랜스포머) 모델
    }

    # --- 모든 모델 학습 및 평가 루프 ---
    rows = []  # 평가 지표 결과를 저장할 리스트 초기화
    predictions = []  # 테스트 예측 결과를 저장할 리스트 초기화
    fitted = {}  # 학습 완료된 모델을 저장할 딕셔너리(dictionary) 초기화
    model_groups = {}  # 모델별 그룹(ML/DL) 매핑을 저장할 딕셔너리(dictionary) 초기화
    for name, (group, model) in models.items():  # 각 모델을 순회하며 학습 및 평가 수행
        model.fit(X_train, y_train)  # 현재 모델을 학습 데이터로 fit(학습)
        fitted[name] = model  # 학습된 모델을 fitted 딕셔너리(dictionary)에 저장
        model_groups[name] = group  # 모델 그룹 정보를 딕셔너리(dictionary)에 저장
        # --- 각 데이터 분할(Split)에 대해 평가 수행 ---
        for split, X_part, y_part in [  # Train(학습), Validation(검증), Test(테스트) 세트를 순회
            ("Train", X_train, y_train),  # Train(학습) 데이터 세트
            ("Validation", X_val, y_val),  # Validation(검증) 데이터 세트
            ("Test", X_test, y_test),  # Test(테스트) 데이터 세트
        ]:
            pred = model.predict(X_part)  # 현재 모델로 해당 분할 데이터에 대해 예측 수행
            rows.append(  # 평가 지표를 딕셔너리(dictionary)로 생성하여 rows 리스트에 추가
                {
                    "Model": name,  # 모델 이름
                    "Group": group,  # 모델 그룹 (Machine Learning / Deep Learning)
                    "Split": split,  # 데이터 분할 유형 (Train/Validation/Test)
                    "RMSE": float(np.sqrt(mean_squared_error(y_part, pred))),  # RMSE(평균 제곱근 오차) 계산
                    "MAE": float(mean_absolute_error(y_part, pred)),  # MAE(평균 절대 오차) 계산
                    "MSE": float(mean_squared_error(y_part, pred)),  # MSE(평균 제곱 오차) 계산
                    "SMAPE(%)": float(smape(y_part, pred)),  # SMAPE(대칭 평균 절대 백분율 오차) 계산
                    "R2": float(r2_score(y_part, pred)) if len(y_part) > 1 else np.nan,  # R²(결정계수) 계산 (샘플 2개 이상인 경우만)
                }
            )
            if split == "Test":  # Test(테스트) 분할인 경우 예측 결과 상세 저장
                pred_df = info_test.copy()  # 테스트 메타 정보 복사
                pred_df["Model"] = name  # 모델 이름 column(컬럼) 추가
                pred_df["Group"] = group  # 모델 그룹 column(컬럼) 추가
                pred_df["Actual"] = y_part.values  # 실제값 column(컬럼) 추가
                pred_df["Predicted"] = pred  # 예측값 column(컬럼) 추가
                predictions.append(pred_df)  # 예측 결과 DataFrame(데이터프레임)을 리스트에 추가

    # --- 평가 결과 DataFrame(데이터프레임) 생성 및 최적 모델 선정 ---
    metrics = pd.DataFrame(rows)  # 평가 지표 리스트를 DataFrame(데이터프레임)으로 변환
    pred_all = pd.concat(predictions, ignore_index=True)  # 모든 테스트 예측 결과를 하나의 DataFrame(데이터프레임)으로 결합
    best_name = metrics[metrics["Split"] == "Test"].sort_values("RMSE").iloc[0]["Model"]  # Test(테스트) 세트 기준 RMSE가 가장 낮은 최적 모델 이름 선정
    best_model = fitted[best_name]  # 최적 모델 객체를 fitted 딕셔너리(dictionary)에서 추출

    # --- Permutation Importance(순열 중요도) 계산 ---
    perm = permutation_importance(  # 최적 모델의 feature(특성) 중요도를 순열 방식으로 계산
        best_model,  # 평가 대상 최적 모델
        X_test,  # Test(테스트) 입력 데이터
        y_test,  # Test(테스트) 타겟 데이터
        n_repeats=20,  # 순열 반복 횟수 (안정적 추정을 위해 20회)
        random_state=42,  # 재현성을 위한 난수 시드 설정
        scoring="neg_mean_squared_error",  # 평가 기준: 음의 MSE(평균 제곱 오차)
    )
    importance = (  # feature(특성) 중요도 DataFrame(데이터프레임) 생성
        pd.DataFrame({"Feature": X.columns, "Importance": perm.importances_mean})  # feature(특성) 이름과 평균 중요도로 DataFrame(데이터프레임) 생성
        .sort_values("Importance", ascending=False)  # 중요도 내림차순으로 정렬
        .reset_index(drop=True)  # 인덱스 리셋
    )

    # --- 최종 결과를 딕셔너리(dictionary)로 반환 ---
    return {  # 학습 및 평가 결과를 딕셔너리(dictionary)로 묶어 반환(리턴)
        "metrics": metrics,  # 모든 모델의 평가 지표 DataFrame(데이터프레임)
        "predictions": pred_all,  # 모든 모델의 테스트 예측 결과 DataFrame(데이터프레임)
        "best_name": best_name,  # 최적 모델 이름 (문자열)
        "model_groups": model_groups,  # 모델별 그룹 매핑 딕셔너리(dictionary)
        "importance": importance,  # feature(특성) 중요도 DataFrame(데이터프레임)
        "feature_columns": list(X.columns),  # feature(특성) column(컬럼) 이름 리스트
        "X": X,  # 전체 feature matrix(특성 행렬)
        "y": y,  # 전체 타겟 벡터
        "X_train": X_train,  # 학습용 feature matrix(특성 행렬)
        "y_train": y_train,  # 학습용 타겟 벡터
        "X_test": X_test,  # 테스트용 feature matrix(특성 행렬)
        "y_test": y_test,  # 테스트용 타겟 벡터
        "info_test": info_test,  # 테스트 세트 메타 정보 (지역, 용도, 부하지수)
        "models": fitted,  # 학습 완료된 모든 모델 딕셔너리(dictionary)
    }
