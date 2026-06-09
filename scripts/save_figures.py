# ============================================================
# 파일명: save_figures.py
# 설명: Baseline(기준선) 모델 학습 후 다양한 시각화 그래프(RMSE 비교, 실제 vs 예측,
#       잔차 분석, QQ Plot, ROC/AUC, t-SNE, CCA, 특성 중요도, Partial Dependence)를
#       PNG 이미지로 저장하는 스크립트
# ============================================================

import os  # os 모듈 import(임포트) — 파일/디렉토리 경로 및 작업 디렉토리 조작용
import sys  # sys 모듈 import(임포트) — Python 경로(sys.path) 조작용

# --- 프로젝트 루트 경로 설정 블록 ---
# 실행 디렉토리에 관계없이 작업 디렉토리(CWD)를 프로젝트 루트로 강제 고정하고 sys.path 등록
script_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트 파일의 디렉토리 절대경로를 구함
project_root = os.path.abspath(os.path.join(script_dir, ".."))  # 스크립트 디렉토리의 상위(부모) 디렉토리를 프로젝트 루트(root)로 설정
os.chdir(project_root)  # 작업 디렉토리(CWD)를 프로젝트 루트로 변경
if project_root not in sys.path:  # 프로젝트 루트가 sys.path에 없으면
    sys.path.insert(0, project_root)  # sys.path 맨 앞에 프로젝트 루트를 추가하여 모듈 import(임포트) 가능하게 설정

# --- 외부 라이브러리 import(임포트) 블록 ---
import pandas as pd  # pandas(판다스) 라이브러리를 pd로 import(임포트) — 데이터 처리용
import numpy as np  # numpy(넘파이) 라이브러리를 np로 import(임포트) — 수치 연산용
import plotly.express as px  # plotly.express(플로틀리 익스프레스)를 px로 import(임포트) — 간편 시각화용
import plotly.graph_objects as go  # plotly.graph_objects(플로틀리 그래프 오브젝트)를 go로 import(임포트) — 상세 시각화용
import plotly.io as pio  # plotly.io를 pio로 import(임포트) — Plotly(플로틀리) I/O 및 template(템플릿) 설정용
from scipy import stats  # scipy.stats를 import(임포트) — 통계 분석 함수(QQ Plot 등)용
from sklearn.cross_decomposition import CCA  # CCA(정준상관분석) 클래스를 import(임포트)
from sklearn.decomposition import PCA  # PCA(주성분분석) 클래스를 import(임포트)
from sklearn.metrics import auc, roc_curve  # AUC(곡선 아래 면적)와 ROC Curve(ROC 곡선) 함수를 import(임포트)
from sklearn.manifold import TSNE  # t-SNE(t-분산 확률적 이웃 임베딩) 클래스를 import(임포트)
from sklearn.preprocessing import StandardScaler  # StandardScaler(표준화 스케일러) 클래스를 import(임포트)

# --- 유틸리티 모듈 import(임포트) 블록 ---
from utils.data_processing import load_all_data, DEFAULT_DATA_DIR  # 데이터 로딩 함수와 기본 데이터 디렉토리 상수를 import(임포트)
from utils.models import make_feature_matrix, train_models  # Feature Matrix(특성 행렬) 생성 함수와 모델 학습 함수를 import(임포트)

# --- Plotly(플로틀리) 기본 template(템플릿) 설정 ---
pio.templates.default = "plotly"  # Plotly(플로틀리)의 기본 컬러 template(템플릿)을 "plotly"로 강제 설정

# --- Baseline(기준선) 시각화 그래프 저장 함수 ---
def save_baseline_figures():  # baseline(기준선) 시각화를 생성하고 PNG로 저장하는 메인 함수 정의
    print("Loading data...")  # 데이터 로딩 시작 메시지 출력
    final_data, monthly_data, _ = load_all_data(DEFAULT_DATA_DIR)  # 기본 데이터 디렉토리에서 전체/월별 데이터를 로드
    
    print("Training models to generate figures...")  # 시각화용 모델 학습 시작 메시지 출력
    model_state = train_models(final_data.to_json(orient="split"))  # DataFrame(데이터프레임)을 JSON으로 변환 후 모델 학습, 결과를 model_state에 저장
    
    fig_dir = "results/baseline/figures"  # 시각화 결과 저장 디렉토리 경로 설정
    os.makedirs(fig_dir, exist_ok=True)  # 시각화 저장 디렉토리 생성 (이미 존재하면 무시)
    
    # --- 용도별 색상 매핑 정의 ---
    cmap = {"자가용": "#00A699", "사업자용": "#FF5A5F"}  # 자가용은 청록색, 사업자용은 붉은색으로 color map(색상 매핑) 정의
    
    # --- 1. Test RMSE 비교 막대 그래프 생성 ---
    print("Generating Test RMSE comparison...")  # Test RMSE 비교 차트 생성 시작 메시지 출력
    metrics = model_state["metrics"]  # model_state에서 성능 메트릭 DataFrame(데이터프레임) 추출
    test_metrics = metrics[metrics["Split"] == "Test"].sort_values("RMSE")  # Test Split(테스트 분할)만 필터링하고 RMSE 오름차순 정렬
    fig = px.bar(  # Plotly Express(플로틀리 익스프레스) 막대 그래프 생성
        test_metrics,  # 데이터로 test_metrics DataFrame(데이터프레임) 사용
        x="Model",  # x축은 모델명
        y="RMSE",  # y축은 RMSE 값
        color="Group",  # 색상을 그룹별로 구분
        title="Test RMSE 비교",  # 차트 제목 설정
        text_auto=".1f",  # 막대 위에 소수점 1자리 텍스트 자동 표시
        color_discrete_sequence=px.colors.qualitative.Plotly  # Plotly(플로틀리) 기본 정성적 색상 팔레트 사용
    )
    fig.write_image(f"{fig_dir}/01_test_rmse.png", scale=2)  # PNG 이미지로 저장 (해상도 2배 scale(스케일))

    # --- 2. Actual vs Predicted(실제 vs 예측) 산점도 생성 ---
    print("Generating Actual vs Predicted...")  # Actual vs Predicted 차트 생성 시작 메시지 출력
    pred = model_state["predictions"]  # model_state에서 예측 결과 DataFrame(데이터프레임) 추출
    best_pred = pred[pred["Model"] == model_state["best_name"]].copy()  # 최적 모델의 예측 결과만 필터링하여 복사본 생성
    fig = px.scatter(  # Plotly Express(플로틀리 익스프레스) 산점도 생성
        best_pred,  # 데이터로 best_pred DataFrame(데이터프레임) 사용
        x="Actual",  # x축은 실제값
        y="Predicted",  # y축은 예측값
        color="용도",  # 색상을 용도(자가용/사업자용)별로 구분
        hover_name="지역",  # 마우스 hover(호버) 시 지역명 표시
        title="Actual vs Prediction",  # 차트 제목 설정
        color_discrete_map=cmap  # 용도별 색상 매핑 적용
    )
    min_v = min(best_pred["Actual"].min(), best_pred["Predicted"].min())  # 실제값과 예측값의 최솟값 중 더 작은 값 계산
    max_v = max(best_pred["Actual"].max(), best_pred["Predicted"].max())  # 실제값과 예측값의 최댓값 중 더 큰 값 계산
    fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", name="완전 일치", line=dict(color='orange')))  # 완전 일치 기준선(대각선)을 주황색으로 추가
    fig.write_image(f"{fig_dir}/02_actual_vs_predicted.png", scale=2)  # PNG 이미지로 저장 (해상도 2배 scale(스케일))
    
    # --- 3. Residual Plot(잔차 분석 도표) 생성 ---
    print("Generating Residual Plot...")  # Residual Plot(잔차 도표) 생성 시작 메시지 출력
    best_pred["Residual"] = best_pred["Actual"] - best_pred["Predicted"]  # 잔차(Residual) = 실제값 - 예측값 계산
    fig = px.scatter(  # Plotly Express(플로틀리 익스프레스) 산점도 생성
        best_pred,  # 데이터로 best_pred DataFrame(데이터프레임) 사용
        x="Predicted",  # x축은 예측값
        y="Residual",  # y축은 잔차
        color="용도",  # 색상을 용도별로 구분
        hover_name="지역",  # 마우스 hover(호버) 시 지역명 표시
        title="Residual Plot",  # 차트 제목 설정
        color_discrete_map=cmap  # 용도별 색상 매핑 적용
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black")  # y=0 위치에 검은색 점선 수평선 추가 (잔차 0 기준선)
    fig.write_image(f"{fig_dir}/03_residual_plot.png", scale=2)  # PNG 이미지로 저장 (해상도 2배 scale(스케일))
    
    # --- 4. QQ Plot(분위수-분위수 도표) 생성 ---
    print("Generating QQ Plot...")  # QQ Plot 생성 시작 메시지 출력
    qq = stats.probplot(best_pred["Residual"], dist="norm")  # 잔차의 정규분포 적합도 검정을 위한 probplot(확률 도표) 계산
    qq_df = pd.DataFrame({"Theoretical": qq[0][0], "Ordered residual": qq[0][1]})  # 이론적 분위수와 정렬된 잔차로 DataFrame(데이터프레임) 생성
    fig = px.scatter(qq_df, x="Theoretical", y="Ordered residual", title="QQ Plot")  # QQ Plot 산점도 생성
    fig.update_traces(marker=dict(color="#636EFA"))  # marker(마커) 색상을 파란색(#636EFA)으로 설정
    fig.write_image(f"{fig_dir}/04_qq_plot.png", scale=2)  # PNG 이미지로 저장 (해상도 2배 scale(스케일))
    
    # --- 5. ROC/AUC(수신자 조작 특성 곡선 / 곡선 아래 면적) 생성 ---
    print("Generating ROC/AUC...")  # ROC/AUC 차트 생성 시작 메시지 출력
    threshold = model_state["y"].quantile(0.7)  # 타겟 변수의 70% quantile(분위수)을 threshold(임계값)으로 설정
    roc_fig = go.Figure()  # Plotly(플로틀리) 빈 Figure(그림) 객체 생성
    y_test_binary = (model_state["y_test"] >= threshold).astype(int)  # 테스트 타겟값을 threshold(임계값) 기준으로 이진(binary) 분류로 변환
    colors = px.colors.qualitative.D3  # D3 정성적 색상 팔레트를 colors에 할당
    for i, (name, model) in enumerate(model_state["models"].items()):  # 모든 모델에 대해 순회 (index(인덱스)와 모델명, 모델 객체)
        score = model.predict(model_state["X_test"])  # 테스트 데이터에 대한 예측 점수 계산
        fpr, tpr, _ = roc_curve(y_test_binary, score)  # FPR(위양성률), TPR(진양성률) 계산
        auc_score = auc(fpr, tpr)  # AUC(곡선 아래 면적) 점수 계산
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} ({auc_score:.3f})", line=dict(color=colors[i % len(colors)])))  # 각 모델의 ROC Curve(ROC 곡선)를 trace(트레이스)로 추가
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="black")))  # Random(무작위) 분류기 기준선(대각선)을 점선으로 추가
    roc_fig.update_layout(title="ROC/AUC: 고부하 위험지역 분류 성능")  # 차트 제목 업데이트
    roc_fig.write_image(f"{fig_dir}/05_roc_auc.png", scale=2)  # PNG 이미지로 저장 (해상도 2배 scale(스케일))
    
    # --- 6. t-SNE(t-분산 확률적 이웃 임베딩) 시각화 생성 ---
    print("Generating t-SNE...")  # t-SNE 차트 생성 시작 메시지 출력
    X_embed = model_state["X"].copy()  # 전체 Feature Matrix(특성 행렬)의 복사본 생성
    y_embed = model_state["y"]  # 타겟 변수(y) 참조
    scaled_embed = StandardScaler().fit_transform(X_embed)  # StandardScaler(표준화 스케일러)로 특성을 표준화(평균 0, 분산 1)
    perplexity = max(5, min(20, len(X_embed) // 4))  # t-SNE perplexity(혼잡도) 값을 데이터 크기에 맞게 동적 설정 (5~20 범위)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", learning_rate="auto")  # t-SNE 객체 생성 (2차원, PCA 초기화, 자동 학습률)
    tsne_xy = tsne.fit_transform(scaled_embed)  # t-SNE 변환 수행하여 2차원 좌표 생성
    embed_df = final_data[["지역", "용도"]].copy()  # 지역과 용도 컬럼만 추출하여 embed(임베딩) DataFrame(데이터프레임) 생성
    embed_df["tSNE-1"] = tsne_xy[:, 0]  # t-SNE 1번째 차원 좌표를 컬럼으로 추가
    embed_df["tSNE-2"] = tsne_xy[:, 1]  # t-SNE 2번째 차원 좌표를 컬럼으로 추가
    embed_df["고부하"] = np.where(y_embed.values >= y_embed.quantile(0.7), "고부하", "일반")  # 70% quantile(분위수) 이상이면 "고부하", 아니면 "일반"으로 라벨 부여
    fig = px.scatter(  # Plotly Express(플로틀리 익스프레스) 산점도 생성
        embed_df,  # 데이터로 embed_df DataFrame(데이터프레임) 사용
        x="tSNE-1",  # x축은 t-SNE 1번째 차원
        y="tSNE-2",  # y축은 t-SNE 2번째 차원
        color="고부하",  # 색상을 고부하/일반으로 구분
        symbol="용도",  # 심볼 모양을 용도별로 구분
        hover_name="지역",  # 마우스 hover(호버) 시 지역명 표시
        title="t-SNE",  # 차트 제목 설정
        color_discrete_map={"고부하": "#FF5A5F", "일반": "#00A699"}  # 고부하는 붉은색, 일반은 청록색으로 색상 매핑
    )
    fig.write_image(f"{fig_dir}/06_tsne.png", scale=2)  # PNG 이미지로 저장 (해상도 2배 scale(스케일))
    
    # --- 7. CCA(정준상관분석) 시각화 생성 ---
    print("Generating CCA...")  # CCA 차트 생성 시작 메시지 출력
    cca_x = StandardScaler().fit_transform(final_data[["전기차_전체대수", "전체_충전기대수", "인프라_부하지수"]])  # 인프라 관련 특성 3개를 표준화
    cca_y = StandardScaler().fit_transform(final_data[["전력_부하지수"]])  # 타겟 변수(전력_부하지수)를 표준화
    cca = CCA(n_components=1)  # CCA(정준상관분석) 객체 생성 (1개 정준 성분)
    x_c, y_c = cca.fit_transform(cca_x, cca_y)  # CCA 학습 및 변환 수행하여 정준 점수(canonical score) 계산
    cca_df = pd.DataFrame({"CCA_X": x_c[:, 0], "CCA_Y": y_c[:, 0], "용도": final_data["용도"], "지역": final_data["지역"]})  # CCA 결과를 DataFrame(데이터프레임)으로 구성
    fig = px.scatter(  # Plotly Express(플로틀리 익스프레스) 산점도 생성
        cca_df,  # 데이터로 cca_df DataFrame(데이터프레임) 사용
        x="CCA_X",  # x축은 X측 정준 점수
        y="CCA_Y",  # y축은 Y측 정준 점수
        color="용도",  # 색상을 용도별로 구분
        hover_name="지역",  # 마우스 hover(호버) 시 지역명 표시
        title="CCA canonical score",  # 차트 제목 설정
        color_discrete_map=cmap  # 용도별 색상 매핑 적용
    )
    fig.write_image(f"{fig_dir}/07_cca.png", scale=2)  # PNG 이미지로 저장 (해상도 2배 scale(스케일))
    
    # --- 8. Permutation Importance(순열 중요도) 막대 그래프 생성 ---
    print("Generating Permutation Importance...")  # Permutation Importance(순열 중요도) 차트 생성 시작 메시지 출력
    importance = model_state["importance"].head(12)  # 상위 12개 Feature(특성)의 중요도 추출
    fig = px.bar(  # Plotly Express(플로틀리 익스프레스) 수평 막대 그래프 생성
        importance.sort_values("Importance"),  # 중요도 기준으로 오름차순 정렬
        x="Importance",  # x축은 중요도 값
        y="Feature",  # y축은 Feature(특성)명
        orientation="h",  # 수평(horizontal) 막대 그래프로 설정
        title="Permutation importance",  # 차트 제목 설정
        color="Importance",  # 색상을 중요도 값에 따라 연속적으로 구분
        color_continuous_scale="Viridis"  # Viridis(비리디스) 연속 색상 스케일 사용
    )
    fig.write_image(f"{fig_dir}/08_importance.png", scale=2)  # PNG 이미지로 저장 (해상도 2배 scale(스케일))
    
    # --- 9. Partial Dependence(부분 의존도) 라인 차트 생성 ---
    print("Generating Partial Dependence...")  # Partial Dependence(부분 의존도) 차트 생성 시작 메시지 출력
    selected_feature = model_state["feature_columns"][0]  # 첫 번째 Feature(특성)을 Partial Dependence(부분 의존도) 분석 대상으로 선택
    best_model = model_state["models"][model_state["best_name"]]  # 최적 모델 객체를 추출
    X_all = model_state["X"].copy()  # 전체 Feature Matrix(특성 행렬)의 복사본 생성
    grid = np.linspace(X_all[selected_feature].quantile(0.05), X_all[selected_feature].quantile(0.95), 30)  # 선택 특성의 5%~95% quantile(분위수) 범위를 30개 grid(격자) 포인트로 분할
    pd_rows = []  # Partial Dependence(부분 의존도) 결과를 저장할 빈 list(리스트) 초기화
    for value in grid:  # 각 grid(격자) 포인트 값에 대해 순회
        X_tmp = X_all.copy()  # 전체 데이터의 복사본 생성
        X_tmp[selected_feature] = value  # 선택 특성 값을 현재 grid(격자) 포인트 값으로 고정
        pd_rows.append({"Feature value": value, "Mean prediction": best_model.predict(X_tmp).mean()})  # 예측값의 평균을 계산하여 결과 list(리스트)에 추가
    fig = px.line(pd.DataFrame(pd_rows), x="Feature value", y="Mean prediction", markers=True, title=f"Partial dependence: {selected_feature}")  # Partial Dependence(부분 의존도) 라인 차트 생성
    fig.update_traces(line=dict(color="#EF553B"), marker=dict(color="#EF553B"))  # 선과 marker(마커) 색상을 붉은색(#EF553B)으로 설정
    fig.write_image(f"{fig_dir}/09_partial_dependence.png", scale=2)  # PNG 이미지로 저장 (해상도 2배 scale(스케일))
    
    print("All colored figures saved successfully to results/baseline/figures/")  # 모든 시각화 저장 완료 메시지 출력

# --- 스크립트 직접 실행 시 진입점 ---
if __name__ == "__main__":  # 이 파일이 직접 실행될 때만 아래 코드 실행
    save_baseline_figures()  # baseline(기준선) 시각화 저장 함수 호출
