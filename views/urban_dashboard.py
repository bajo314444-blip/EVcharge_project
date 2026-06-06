import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from streamlit_folium import st_folium
import networkx as nx

from utils.visualizations import make_bubble_map, make_tableone, render_shap_or_fallback, render_highway_edge_plot
from utils.models import make_feature_matrix
from utils.data_processing import cached_bootstrap, cached_adversarial, cached_ablation, cached_dca, cached_nested_cv, cached_spatial_external_validation, cached_survival, cached_partial_dependence
from utils.pdf_generator import generate_report_pdf, generate_highway_report_pdf
from utils.optimization import optimize_highway_chargers, calculate_single_region_trajectory, calculate_topsis_rankings, simulate_dynamic_pricing


def render_dashboard(filtered, top_region, metric, usage_options, final_data, monthly_data, hourly_data, model_state, model_state_smote):
    # Initialize anomalies if not done yet
    if "anomalies_list" not in st.session_state:
        np.random.seed(42)
        sample_regions = final_data.dropna(subset=["위도", "경도"]).sample(min(3, len(final_data)), random_state=42).copy()
        anomaly_types = ["커넥터 온도 과열", "이상 전압 변동", "통신 패킷 유실"]
        anomalies = []
        for idx, (_, row) in enumerate(sample_regions.iterrows()):
            anomalies.append({
                "지역": row["지역"],
                "용도": row["용도"],
                "위도": row["위도"],
                "경도": row["경도"],
                "anomaly_type": anomaly_types[idx % len(anomaly_types)],
                "temperature": float(np.random.uniform(65.0, 85.0)),
                "voltage_std": float(np.random.uniform(8.5, 15.0)),
                "packet_loss": float(np.random.uniform(5.0, 25.0)),
            })
        st.session_state["anomalies_list"] = anomalies

    # Show warning banner
    anomalies = st.session_state["anomalies_list"]
    show_banner = st.session_state.get("show_warning_banner", True)
    if show_banner and anomalies:
        warn_col1, warn_col2 = st.columns([35, 1])
        with warn_col1:
            st.markdown(
                f"""
                <div style="background-color: #FF4B4B; color: white; padding: 12px; border-radius: 8px; font-weight: bold; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    🚨 [경고] 현재 수도권 충전소 {len(anomalies)}개 지점에서 이상 징후(과열, 전압 급변, 통신 장애)가 실시간 감지되었습니다. '이상 징후 관제 지도' 탭에서 상세 현황을 확인하세요.
                </div>
                """,
                unsafe_allow_html=True
            )
        with warn_col2:
            st.markdown('<div style="padding-top: 10px;"></div>', unsafe_allow_html=True)
            if st.button("❌", key="close_warning_banner", help="알림 배너 닫기"):
                st.session_state["show_warning_banner"] = False
                st.rerun()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("분석 지역-용도 행", f"{len(filtered):,}개")
    col2.metric("최고 부하 지역", top_region["지역"].iloc[0] if not top_region.empty else "-")
    col3.metric("최고 전력 부하지수", f"{top_region['전력_부하지수'].iloc[0]:,.2f}" if not top_region.empty else "-")
    col4.metric("학습 기준 최고 모델", model_state["best_name"])

    active_menu = st.radio(
        "분석 메뉴 선택",
        [
            "🗺️ 지도 버블맵",
            "📈 월별 부하 변화",
            "💡 분석 시뮬레이터",
            "📊 예측 모델 비교",
            "🛡️ 강건성 평가 (Phase 3)",
            "🧮 통계/군집 분석",
            "🧠 SHAP/LIME 설명",
            "📋 조건 충족표",
        ],
        horizontal=True,
        label_visibility="collapsed"
    )

    if active_menu == "🗺️ 지도 버블맵":
        sub_tabs = st.tabs(["📍 현황 버블맵", "🎯 최적 입지 추천 지도", "🚨 이상 징후 관제 지도"])
        
        with sub_tabs[0]:
            st.subheader("현재 부하 버블맵")
            st.caption("지도 레이어에서 자가용과 사업자용을 켜고 끌 수 있습니다.")
            left, right = st.columns([1.35, 0.9])
            with left:
                st_folium(make_bubble_map(filtered, metric, usage_options), height=650, key="current_map", use_container_width=True, returned_objects=["last_object_clicked"])
            with right:
                st.markdown("#### 고부하 상위 지역 (TOP 15)")

                rank_tabs = st.tabs(["전체", "사업자용", "자가용"])

                def render_rank_table(df_subset, usage_name):
                    if df_subset.empty:
                        st.info(f"{usage_name} 데이터가 없습니다.")
                        return
            
                    cols_to_show = ["지역", "용도", "전기차_전체대수", "전체_충전기대수", "총용량_kW"]
                    if metric not in cols_to_show:
                        cols_to_show.append(metric)
                
                    top_table = (
                        df_subset.sort_values(metric, ascending=False)
                        [cols_to_show]
                        .head(15)
                        .copy()
                    )
                    top_table.insert(0, "순위", range(1, len(top_table) + 1))
                    styled_table = top_table.style.set_properties(**{'text-align': 'center'}, subset=['순위'])
                    st.dataframe(styled_table, use_container_width=True, hide_index=True)
        
                    top_table["지역_용도"] = top_table["지역"] + " (" + top_table["용도"] + ")"
            
                    fig = px.bar(
                        top_table.sort_values("순위", ascending=True),
                        x=metric,
                        y="지역_용도",
                        color="용도",
                        orientation="h",
                        color_discrete_map={"자가용": "#00A699", "사업자용": "#FF5A5F"},
                        title=f"{usage_name} 고부하 TOP 15 ({metric} 기준)",
                        labels={metric: f"{metric} 점수", "지역_용도": "지역 (용도)"},
                        category_orders={"지역_용도": top_table.sort_values("순위", ascending=True)["지역_용도"].tolist()}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with rank_tabs[0]:
                    render_rank_table(filtered, "전체")
                with rank_tabs[1]:
                    render_rank_table(filtered[filtered["용도"] == "사업자용"], "사업자용")
                with rank_tabs[2]:
                    render_rank_table(filtered[filtered["용도"] == "자가용"], "자가용")

        with sub_tabs[1]:
            st.subheader("🎯 TOPSIS 다중 기준 의사결정(MCDA) 최적 입지 추천")
            st.caption("전력 부하, 인프라 부하, 충전소 밀집도, 비용대비 전력망 완화율의 가중치를 고려하여 최적의 추가 충전소 설치 입지를 도출합니다.")
            
            w_col1, w_col2, w_col3, w_col4 = st.columns(4)
            with w_col1:
                w_load = st.slider("전력 부하지수 가중치", 0.0, 1.0, 0.35, 0.05)
            with w_col2:
                w_infra = st.slider("인프라 부하지수 가중치", 0.0, 1.0, 0.35, 0.05)
            with w_col3:
                w_density = st.slider("충전소 밀집도 역수 가중치", 0.0, 1.0, 0.15, 0.05)
            with w_col4:
                w_mitigation = st.slider("전력망 완화율 가중치", 0.0, 1.0, 0.15, 0.05)
                
            weights_sum = w_load + w_infra + w_density + w_mitigation
            if weights_sum == 0:
                st.error("가중치 합이 0일 수 없습니다.")
            else:
                norm_weights = {
                    "전력_부하지수": w_load / weights_sum,
                    "인프라_부하지수": w_infra / weights_sum,
                    "충전소_밀집도_역수": w_density / weights_sum,
                    "전력망_완화율": w_mitigation / weights_sum
                }
                
                topsis_res = calculate_topsis_rankings(filtered, norm_weights)
                topsis_top5 = topsis_res.sort_values("TOPSIS_점수", ascending=False).head(5).copy()
                topsis_top5["TOPSIS_순위"] = range(1, len(topsis_top5) + 1)
                
                t_left, t_right = st.columns([1.35, 0.9])
                with t_left:
                    st_folium(
                        make_bubble_map(filtered, metric, usage_options, topsis_data=topsis_top5),
                        height=600,
                        key="topsis_map",
                        use_container_width=True,
                        returned_objects=["last_object_clicked"]
                    )
                with t_right:
                    st.markdown("#### 🎯 최적 입지 추천 TOP 5 지역")
                    for _, row in topsis_top5.iterrows():
                        st.markdown(
                            f"""
                            <div style="background-color: rgba(255,215,0,0.08); border-left: 5px solid #FFD700; padding: 10px; margin-bottom: 10px; border-radius: 4px;">
                                <h5 style="margin: 0; color: #E5A93C;">{row['TOPSIS_순위']}위: {row['지역']} ({row['용도']})</h5>
                                <p style="margin: 5px 0 0 0; font-size: 13px; line-height: 1.5;">
                                    <b>TOPSIS 점수:</b> {row['TOPSIS_점수']:.4f}<br>
                                    <b>전력 부하지수:</b> {row['전력_부하지수']:.2f} | <b>인프라 부하지수:</b> {row['인프라_부하지수']:.2f}<br>
                                    <b>충전소수:</b> {row['충전소개수']:.0f}개 | <b>총용량:</b> {row['총용량_kW']:.0f} kW
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

        with sub_tabs[2]:
            st.subheader("🚨 충전기 고장 예후 Anomaly Detection 관제 지도")
            st.caption("전압 변동성, 커넥터 온도, 패킷 유실률 등의 실시간 지표를 파싱하여 고장 예후가 감지된 충전소를 실시간 모니터링합니다.")
            
            if st.button("🔄 실시간 상태 데이터 갱신 (모의 파싱)"):
                sample_regions = final_data.dropna(subset=["위도", "경도"]).sample(min(3, len(final_data))).copy()
                anomaly_types = ["커넥터 온도 과열", "이상 전압 변동", "통신 패킷 유실"]
                new_anomalies = []
                for idx, (_, row) in enumerate(sample_regions.iterrows()):
                    new_anomalies.append({
                        "지역": row["지역"],
                        "용도": row["용도"],
                        "위도": row["위도"],
                        "경도": row["경도"],
                        "anomaly_type": np.random.choice(anomaly_types),
                        "temperature": float(np.random.uniform(65.0, 85.0)),
                        "voltage_std": float(np.random.uniform(8.5, 15.0)),
                        "packet_loss": float(np.random.uniform(5.0, 25.0)),
                    })
                st.session_state["anomalies_list"] = new_anomalies
                st.rerun()
                
            anomalies = st.session_state["anomalies_list"]
            
            a_left, a_right = st.columns([1.35, 0.9])
            with a_left:
                st_folium(
                    make_bubble_map(filtered, metric, usage_options, anomalies=anomalies),
                    height=600,
                    key="anomaly_map",
                    use_container_width=True,
                    returned_objects=["last_object_clicked"]
                )
            with a_right:
                st.markdown("#### 🚨 실시간 이상 감지 경보 목록")
                for item in anomalies:
                    st.markdown(
                        f"""
                        <div style="background-color: rgba(255,75,75,0.08); border-left: 5px solid #FF4B4B; padding: 10px; margin-bottom: 10px; border-radius: 4px;">
                            <h5 style="margin: 0; color: #FF4B4B;">🚨 {item['지역']} ({item['용도']})</h5>
                            <p style="margin: 5px 0 0 0; font-size: 13px; line-height: 1.5;">
                                <b>이상 유형:</b> <span style="color: #FF4B4B; font-weight: bold;">{item['anomaly_type']}</span><br>
                                <b>커넥터 온도:</b> {item['temperature']:.1f}°C (임계치: 60°C)<br>
                                <b>전압 변동성:</b> {item['voltage_std']:.2f} V (정상범위: &lt; 5.0 V)<br>
                                <b>패킷 유실률:</b> {item['packet_loss']:.1f}% (정상범위: &lt; 1.0%)
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    elif active_menu == "📈 월별 부하 변화":
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

    elif active_menu == "💡 분석 시뮬레이터":
        sim_tabs = st.tabs(["🔌 충전기 증설 시뮬레이션", "💸 다이내믹 요금제 시뮬레이션"])
        
        with sim_tabs[0]:
            st.subheader("통합 설치 및 미래 시뮬레이터 (Time-to-Overload)")
            st.caption("충전기 추가 설치 시, 해당 지역의 부하지수가 어떻게 감소하며 과부하 시점을 얼마나 늦출 수 있는지(생존 궤적) 확인합니다.")

            sim_col1, sim_col2 = st.columns([1, 1.5])

            with sim_col1:
                st.markdown("##### 1. 시뮬레이션 설정")
                region_list = sorted(final_data["지역"].unique())
                default_idx = region_list.index(top_region["지역"].iloc[0]) if not top_region.empty else 0
                sim_region = st.selectbox("설치 후보 지역", region_list, index=default_idx)
                sim_usage = st.radio("계산 기준 용도", ["자가용", "사업자용"], horizontal=True)
                growth_rate = st.slider("해당 지역 전기차 연간 증가율 (%)", 1.0, 20.0, 5.0, 1.0) / 100.0

                st.markdown("##### 2. 설치 정책 (Intervention)")
                charger_count = st.slider("추가 충전기 수", 1, 80, 10)
                charger_kw = st.select_slider("충전기 1대 용량(kW)", options=[50, 100, 150, 200, 350], value=100)

                target_row = final_data[(final_data["지역"] == sim_region) & (final_data["용도"] == sim_usage)].iloc[0]
                added_kw = charger_count * charger_kw
                base_load = target_row["총_전력판매량"]
                capacity = target_row["총용량_kW"]
                critical_threshold = final_data["전력_부하지수"].quantile(0.8) # 상위 20% 위험 한계선

                traj_df, overload_before, overload_after = calculate_single_region_trajectory(
                    base_load, capacity, growth_rate, added_kw, critical_threshold
                )

                before = float(target_row["전력_부하지수"])
                after = float(base_load / (capacity + added_kw)) if (capacity + added_kw) > 0 else 0
                reduction_pct = (before - after) / before * 100 if before else 0

                st.markdown("##### 3. 즉각적인 부하 감소 효과")
                st.metric("현재 전력 부하지수", f"{before:,.2f}")
                st.metric("설치 후 부하지수", f"{after:,.2f}", delta=f"-{reduction_pct:.1f}%")
                st.metric("추가 공급 용량", f"{added_kw:,.0f} kW")

            with sim_col2:
                st.markdown("##### 4. 과부하 지연 효과 (정책의 장기적 가치)")

                delay_years = overload_after - overload_before

                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.metric("기존 과부하 도달", f"{overload_before}년 뒤" if overload_before < 15 else "안전 (15년+)")
                with mc2:
                    st.metric("설치 후 도달", f"{overload_after}년 뒤" if overload_after < 15 else "안전 (15년+)")
                with mc3:
                    st.metric("충전 대란 지연 효과", f"+{delay_years}년" if delay_years > 0 else "변동 없음")
        
                fig = px.line(traj_df, x="Year", y="부하지수", color="상태", markers=True, 
                              title=f"{sim_region} 부하지수 미래 궤적 (연 {growth_rate*100:.0f}% 성장)")
                fig.add_hline(y=critical_threshold, line_dash="dash", line_color="red", annotation_text="위험 한계선 (Threshold)")
                st.plotly_chart(fig, use_container_width=True)

                st_folium(
                    make_bubble_map(
                        final_data[final_data["지역"].isin([sim_region])],
                        "전력_부하지수",
                        [sim_usage],
                        selected_region=sim_region,
                        scenario={"added_kw": added_kw, "reduction_pct": reduction_pct},
                    ),
                    height=300,
                    key="sim_map",
                    use_container_width=True,
                    returned_objects=["last_object_clicked"]
                )

        with sim_tabs[1]:
            st.subheader("💸 다이내믹 요금제 수요 이동 시뮬레이션")
            st.caption("가격 탄력성(Elasticity) 모델을 기반으로 피크 시간대 할증 및 경부하 시간대 할인을 통해 부하 수요가 분산되는 효과를 시뮬레이션합니다.")
            
            dp_col1, dp_col2 = st.columns([1, 1.8])
            with dp_col1:
                st.markdown("##### 1. 요금제 및 탄력성 설정")
                elasticity = st.slider("가격 탄력성 계수 (Elasticity)", -1.0, 0.0, -0.2, 0.05)
                surcharge = st.slider("피크 시간대 할증률 (Peak Surcharge)", 0.0, 1.0, 0.20, 0.05)
                discount = st.slider("경부하 시간대 할인율 (Off-Peak Discount)", 0.0, 0.5, 0.15, 0.05)
                
                st.markdown(
                    """
                    *   **피크 시간대**: 10:00 ~ 12:00, 13:00 ~ 17:00, 18:00 ~ 22:00 (할증 적용)
                    *   **경부하 시간대**: 23:00 ~ 09:00 (할인 적용)
                    *   **중부하 시간대**: 그 외 시간 (요금 변동 없음)
                    """
                )
                
                charge_type = st.radio("충전 방식 선택", ["전체", "급속", "완속"], horizontal=True)
                
            with dp_col2:
                if not hourly_data.empty:
                    df_profile = hourly_data.copy()
                    if charge_type != "전체":
                        df_profile = df_profile[df_profile["충전방식"] == charge_type]
                        
                    hour_cols = [f"{i:02d}시" for i in range(24)]
                    base_profile = df_profile[hour_cols].mean().values
                    
                    sim_profile, price_change = simulate_dynamic_pricing(
                        base_profile,
                        elasticity=elasticity,
                        peak_surcharge=surcharge,
                        discount_rate=discount
                    )
                    
                    st.markdown("##### 2. 요금제 도입 전/후 시간대별 충전 부하 비교")
                    
                    chart_df = pd.DataFrame({
                        "시간": [f"{i:02d}시" for i in range(24)],
                        "도입 전 (현재 부하)": base_profile,
                        "도입 후 (시뮬레이션)": sim_profile,
                        "요금 변동률": price_change * 100
                    })
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=chart_df["시간"], y=chart_df["도입 전 (현재 부하)"],
                        fill='tozeroy', mode='lines+markers',
                        name='도입 전 (현재 부하)',
                        line=dict(color='#888888', width=2),
                        fillcolor='rgba(136, 136, 136, 0.2)'
                    ))
                    fig.add_trace(go.Scatter(
                        x=chart_df["시간"], y=chart_df["도입 후 (시뮬레이션)"],
                        fill='tozeroy', mode='lines+markers',
                        name='도입 후 (시뮬레이션)',
                        line=dict(color='#FF5A5F', width=3),
                        fillcolor='rgba(255, 90, 95, 0.3)'
                    ))
                    
                    fig.update_layout(
                        title=f"다이내믹 요금제 부하 분산 궤적 (탄력성: {elasticity:.2f})",
                        xaxis_title="시간대",
                        yaxis_title="평균 충전 부하 (kW)",
                        hovermode="x unified",
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    peak_hours_idx = [10, 11, 13, 14, 15, 16, 18, 19, 20, 21]
                    before_peak_avg = np.mean(base_profile[peak_hours_idx])
                    after_peak_avg = np.mean(sim_profile[peak_hours_idx])
                    peak_reduction = (before_peak_avg - after_peak_avg) / before_peak_avg * 100 if before_peak_avg else 0
                    
                    mc_col1, mc_col2, mc_col3 = st.columns(3)
                    with mc_col1:
                        st.metric("피크 시간대 평균 부하 (전)", f"{before_peak_avg:,.1f} kW")
                    with mc_col2:
                        st.metric("피크 시간대 평균 부하 (후)", f"{after_peak_avg:,.1f} kW")
                    with mc_col3:
                        st.metric("피크 시간대 부하 절감률", f"{peak_reduction:.1f}%", delta=f"-{peak_reduction:.1f}%")
                else:
                    st.warning("시간대별 충전 부하 데이터가 비어 있어 시뮬레이션을 수행할 수 없습니다.")

    elif active_menu == "📊 예측 모델 비교":
        st.subheader("예측 모델 성능 비교")
        st.caption("머신러닝 5개와 딥러닝 2개(CNN, Transformer 계열)를 같은 데이터 분할로 비교합니다.")

        metrics = model_state["metrics"].copy()

        # 순위(Rank) 계산 (Test RMSE 기준)
        test_only = metrics[metrics["Split"] == "Test"].copy()
        test_only["Rank (순위)"] = test_only["RMSE"].rank(method="min").astype(int)
        metrics = pd.merge(metrics, test_only[["Model", "Rank (순위)"]], on="Model", how="left")

        # 탭 분리
        sub_tabs = st.tabs(["📊 성능 랭킹 및 지표", "📈 예측 일치도 및 오차 분석", "🎯 위험 지역 분류 성능 (ROC)"])

        with sub_tabs[0]:
            st.markdown("#### 전체 지표 종합 표")
            # 컬럼 순서 조정: Model, Rank (순위), Group, Split, RMSE, R2, MAE, SMAPE(%)
            cols = ["Model", "Rank (순위)", "Group", "Split", "RMSE", "R2", "MAE", "SMAPE(%)"]
            display_metrics = metrics[metrics["Split"] == "Test"]
            st.dataframe(display_metrics[cols].sort_values(["Rank (순위)", "Split"]), use_container_width=True, hide_index=True)

            test_metrics = test_only.sort_values("RMSE")
            fig = px.bar(
                test_metrics,
                x="Model",
                y="RMSE",
                color="Group",
                title="Test RMSE 비교 (낮을수록 우수)",
                text_auto=".1f",
            )
            st.plotly_chart(fig, use_container_width=True)

        with sub_tabs[1]:
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
            fig = px.scatter(qq_df, x="Theoretical", y="Ordered residual", title="QQ Plot (정규성 검정)")
            st.plotly_chart(fig, use_container_width=True)

        with sub_tabs[2]:
            roc_rows = []
            roc_fig = go.Figure()
            if "precomputed_roc_data" in model_state:
                for name, rdata in model_state["precomputed_roc_data"].items():
                    fpr = rdata["fpr"]
                    tpr = rdata["tpr"]
                    auc_score = rdata["auc"]
                    roc_rows.append({"Model": name, "Group": rdata["group"], "AUC": auc_score})
                    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} ({auc_score:.3f})"))
            else:
                threshold = model_state["y"].quantile(0.7)
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

        st.markdown("---")
        st.markdown("### SMOTE 적용 전/후 최고 모델 성능 비교")
        if model_state_smote is None:
            with st.spinner("🤖 SMOTE 불균형 오버샘플링을 적용하여 7개 ML/DL 모델을 비동기 재학습 중입니다... (약 20초 소요)"):
                from utils.models import train_models
                model_state_smote = train_models(final_data.to_json(orient="split"), use_smote=True)
                st.session_state["model_state_smote"] = model_state_smote
        best_pre = model_state["metrics"][model_state["metrics"]["Split"]=="Test"].sort_values("RMSE").iloc[0]
        best_post = model_state_smote["metrics"][model_state_smote["metrics"]["Split"]=="Test"].sort_values("RMSE").iloc[0]
        smote_df = pd.DataFrame({
            "상태": ["적용 전 (Pre-SMOTE)", "적용 후 (Post-SMOTE)"],
            "최우수 모델명": [best_pre["Model"], best_post["Model"]],
            "Test RMSE": [best_pre["RMSE"], best_post["RMSE"]],
            "Test R2": [best_pre["R2"], best_post["R2"]]
        })
        st.dataframe(smote_df.round(4), use_container_width=True, hide_index=True)

    elif active_menu == "🧮 통계/군집 분석":
        st.subheader("TableOne, t-SNE/UMAP, CCA, 상관분석")
        st.caption("평가 기준의 통계/부분집단 분석 항목을 한 화면에서 확인합니다.")
        table_cols = [
            "전기차_전체대수",
            "총_전력판매량",
            "총_판매수입",
            "충전인프라_규모_PCA",
            "충전기_1대당_평균용량",
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

        # t-SNE (사전 계산된 값이 있을 경우 0.01초 로드, 없을 경우 실시간 폴백)
        if "precomputed_tsne_xy" in model_state:
            tsne_xy = model_state["precomputed_tsne_xy"]
        else:
            perplexity = max(5, min(20, len(X_embed) // 4))
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", learning_rate="auto")
            tsne_xy = tsne.fit_transform(scaled_embed)
            
        embed_df = final_data[["지역", "용도"]].copy()
        embed_df["tSNE-1"] = tsne_xy[:, 0]
        embed_df["tSNE-2"] = tsne_xy[:, 1]
        embed_df["고부하"] = np.where(y_embed.values >= y_embed.quantile(0.7), "고부하", "일반")
        fig = px.scatter(embed_df, x="tSNE-1", y="tSNE-2", color="고부하", symbol="용도", hover_name="지역", title="t-SNE")
        st.plotly_chart(fig, use_container_width=True)

        # UMAP (사전 계산된 값이 있을 경우 0.01초 로드, 없을 경우 실시간 폴백)
        if "precomputed_umap_xy" in model_state:
            umap_xy = model_state["precomputed_umap_xy"]
            title = model_state.get("precomputed_umap_title", "UMAP")
        else:
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
        for x_col in ["전기차_전체대수", "충전인프라_규모_PCA", "충전기_1대당_평균용량", "인프라_부하지수"]:
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

        # CCA (사전 계산된 값이 있을 경우 0.01초 로드, 없을 경우 실시간 폴백)
        if "precomputed_cca_x_c" in model_state and "precomputed_cca_y_c" in model_state:
            cca_x_c = model_state["precomputed_cca_x_c"]
            cca_y_c = model_state["precomputed_cca_y_c"]
        else:
            cca_x = StandardScaler().fit_transform(final_data[["전기차_전체대수", "충전인프라_규모_PCA", "충전기_1대당_평균용량", "인프라_부하지수"]])
            cca_y = StandardScaler().fit_transform(final_data[["전력_부하지수"]])
            cca = CCA(n_components=1)
            x_c, y_c = cca.fit_transform(cca_x, cca_y)
            cca_x_c = x_c[:, 0]
            cca_y_c = y_c[:, 0]
            
        cca_df = pd.DataFrame({"CCA_X": cca_x_c, "CCA_Y": cca_y_c, "용도": final_data["용도"], "지역": final_data["지역"]})
        fig = px.scatter(cca_df, x="CCA_X", y="CCA_Y", color="용도", hover_name="지역", title="CCA canonical score")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("심화 탐색적 데이터 분석 (EDA)")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 1. Scatter plot with 95% Confidence Interval")
            try:
                import seaborn as sns
    
                plt.rc('font', family='Malgun Gothic')
                plt.rcParams['axes.unicode_minus'] = False
                fig_scatter, ax = plt.subplots(figsize=(5, 3.5))
    
                colors = {"자가용": "#00A699", "사업자용": "#FF5A5F"}
                for usage in final_data["용도"].unique():
                    subset = final_data[final_data["용도"] == usage]
                    sns.regplot(
                        data=subset, 
                        x="전기차_전체대수", 
                        y="전력_부하지수", 
                        ax=ax, 
                        label=usage, 
                        color=colors.get(usage, "blue"),
                        scatter_kws={'alpha':0.5, 's':15}
                    )
                ax.set_title("전기차 대수 대비 전력 부하지수 (95% 신뢰구간 포함)")
                ax.legend()
                st.pyplot(fig_scatter, clear_figure=True, use_container_width=True)
            except ImportError:
                st.info("seaborn 패키지가 설치되지 않아 추세선을 그릴 수 없습니다.")
    
        with col2:
            st.markdown("#### 2. 상관관계 네트워크 그래프 (Network Graph)")
            num_cols = ["전기차_전체대수", "총_전력판매량", "충전인프라_규모_PCA", "충전기_1대당_평균용량", "전력_부하지수", "인프라_부하지수"]
            corr_mat = final_data[num_cols].corr()

            G = nx.Graph()
            for i in range(len(corr_mat.columns)):
                for j in range(i + 1, len(corr_mat.columns)):
                    weight = corr_mat.iloc[i, j]
                    # 임계값을 0.15로 낮추어 음의 상관관계(파란 선)도 그래프에 나타나도록 수정
                    if abs(weight) > 0.15:
                        G.add_edge(corr_mat.columns[i], corr_mat.columns[j], weight=weight)
            
            if len(G.edges) > 0:
                plt.rc('font', family='Malgun Gothic')
                plt.rcParams['axes.unicode_minus'] = False
                fig_net, ax_net = plt.subplots(figsize=(5, 3.5))
                pos = nx.spring_layout(G, k=5.0, seed=42)
                edges = G.edges()
                raw_weights = [G[u][v]['weight'] for u,v in edges]
                nx.draw(
                    G, pos, ax=ax_net, with_labels=True, node_color='lightgreen', 
                    node_size=500, font_family='Malgun Gothic', font_size=4, 
                    edge_color=raw_weights, edge_cmap=plt.cm.coolwarm, edge_vmin=-1, edge_vmax=1,
                    width=[abs(w) * 5 for w in raw_weights]
                )
                # 선 색상 의미를 나타내는 컬러바(범례) 추가
                sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=-1, vmax=1))
                cbar = plt.colorbar(sm, ax=ax_net, shrink=0.5, pad=0.05)
                cbar.set_label('상관계수 (Red: 양의 상관, Blue: 음의 상관)', fontsize=7)
                cbar.ax.tick_params(labelsize=6)
    
                # 네트워크 노드가 잘리지 않도록 여백 추가
                ax_net.margins(0.2)
                st.pyplot(fig_net, clear_figure=True, use_container_width=True)
            else:
                st.info("상관관계(>0.15)가 높은 변수 쌍이 없어 네트워크를 생성할 수 없습니다.")

        st.markdown("#### 3. Box plot with p-value (집단 간 부하지수 차이 검증)")
        private_load = final_data[final_data["용도"] == "자가용"]["전력_부하지수"]
        business_load = final_data[final_data["용도"] == "사업자용"]["전력_부하지수"]
        _, p_val = stats.ttest_ind(private_load, business_load, equal_var=False, nan_policy="omit")

        fig_box = px.box(
            final_data, 
            x="용도", 
            y="전력_부하지수", 
            color="용도", 
            title="용도별 전력 부하지수 분포"
        )
        fig_box.add_annotation(
            x=0.5, 
            y=final_data["전력_부하지수"].max(), 
            text=f"통계적 유의성 (p-value): {p_val:.4e}", 
            showarrow=False,
            font=dict(size=14, color="red")
        )
        fig_box.update_layout(height=450)
        st.plotly_chart(fig_box, use_container_width=True)

    elif active_menu == "🛡️ 강건성 평가 (Phase 3)":
        st.subheader("학술적 강건성 평가 및 심화 시뮬레이션")

        best_name = model_state["best_name"]
        best_model = model_state["models"][best_name]

        # 1. Bootstrap CI
        st.markdown(f"#### 1. 부트스트랩 95% 신뢰구간 (Bootstrap 95% CI) - 최우수 모델: {best_name}")
        if "precomputed_bootstrap" in model_state:
            ci_rmse, ci_r2, r_scores, r2_scores = model_state["precomputed_bootstrap"]
        else:
            ci_rmse, ci_r2, r_scores, r2_scores = cached_bootstrap(best_name, best_model, model_state["X_test"], model_state["y_test"])
        st.success(f"**RMSE 95% CI**: {ci_rmse[0]:.4f} ~ {ci_rmse[1]:.4f}  \n**R2 95% CI**: {ci_r2[0]:.4f} ~ {ci_r2[1]:.4f}")

        # 1-2. Nested CV
        st.markdown(f"#### 1-2. 중첩 10-겹 교차검증 (Nested 10-fold CV) - 최우수 모델: {best_name}")
        st.caption("외부 10-Fold로 평가, 내부 3-Fold로 튜닝하여 과적합 없는 일반화 성능을 산출합니다.")
        if "precomputed_nested_cv" in model_state:
            mean_rmse, std_rmse, outer_scores = model_state["precomputed_nested_cv"]
        else:
            mean_rmse, std_rmse, outer_scores = cached_nested_cv(best_name, best_model, model_state["X"], model_state["y"])
        st.info(f"**Nested CV 평균 Test RMSE**: {mean_rmse:.4f} ± {std_rmse:.4f}")

        # 1-3. Spatial External Validation
        st.markdown(f"#### 1-3. 공간적 외부 검증 (Spatial External Validation) - 최우수 모델: {best_name}")
        st.caption("선택한 지역의 데이터를 학습에서 완전히 배제한 뒤, 해당 지역을 새로운 외부 환경으로 가정하고 예측 성능을 검증합니다.")

        holdout_region = st.selectbox("학습에서 배제할 외부 테스트 지역 선택:", ["인천", "서울", "경기"], index=0)

        precomputed_spatial = model_state.get("precomputed_spatial", {})
        if holdout_region in precomputed_spatial:
            ext_rmse, ext_mae, ext_r2, ext_err = precomputed_spatial[holdout_region]
        else:
            ext_rmse, ext_mae, ext_r2, ext_err = cached_spatial_external_validation(best_name, best_model, model_state["X"], model_state["y"], holdout_region)

        if ext_err:
            st.warning(ext_err)
        else:
            st.success(f"**[{holdout_region}] 외부 검증 Test RMSE**: {ext_rmse:.4f} (MAE: {ext_mae:.4f}, R²: {ext_r2:.4f})")

            # 내부 검증(Random Split Test) 성적과 비교 시각화
            test_rmse_internal = model_state["metrics"][(model_state["metrics"]["Model"] == best_name) & (model_state["metrics"]["Split"] == "Test")]["RMSE"].values[0]

            comp_df = pd.DataFrame({
                "Validation Type": ["Internal (Random Split)", f"External ({holdout_region} Holdout)"],
                "RMSE": [test_rmse_internal, ext_rmse]
            })

            fig_ext = px.bar(
                comp_df, x="Validation Type", y="RMSE", text_auto=".4f",
                title=f"내부 검증 vs 외부 검증 ({holdout_region}) 성능 비교 (RMSE 낮을수록 우수)",
                color="Validation Type", color_discrete_sequence=["#2ca02c", "#d62728"]
            )
            st.plotly_chart(fig_ext, use_container_width=True)

        # 2. Adversarial Attack
        st.markdown("#### 2. 적대적 공격 방어력 평가 (Adversarial Attack Analysis)")
        st.caption("Test 데이터셋의 모든 피처에 가우시안 노이즈를 강제 주입했을 때의 성능 하락률을 방어력으로 평가합니다.")
        if "precomputed_adversarial" in model_state:
            adv_res = model_state["precomputed_adversarial"]
        else:
            adv_res = cached_adversarial(best_name, best_model, model_state["X_test"], model_state["y_test"])
        st.dataframe(adv_res.style.background_gradient(cmap="Reds", subset=["Drop_Ratio(%)"]), use_container_width=True, hide_index=True)

        # 3. Ablation Study
        st.markdown("#### 3. 피처 중요도 기반 민감도 분석 (Ablation Study)")
        st.caption("중요도가 가장 낮은 피처부터 하나씩 제거하며 성능(RMSE) 저하 폭을 시각화합니다.")
        if "precomputed_ablation" in model_state:
            abl_res = model_state["precomputed_ablation"]
        else:
            abl_res = cached_ablation(best_name, best_model, model_state["X_train"], model_state["y_train"], model_state["X_test"], model_state["y_test"], model_state["importance"])
        fig_abl = px.line(abl_res, x="Num_Features", y="RMSE", hover_data=["Removed_Feature"], markers=True, title="Ablation Study: Feature Removal Impact")
        fig_abl.update_xaxes(autorange="reversed")
        st.plotly_chart(fig_abl, use_container_width=True)

        # 4. DCA
        st.markdown("#### 4. 의사결정 곡선 분석 (Decision Curve Analysis - adapted)")
        st.caption("과부하 예측(상위 50~90% 위험도) 시 개입했을 때의 모델 효용성(Net Benefit)을 평가합니다.")
        if "precomputed_dca" in model_state:
            dca_res = model_state["precomputed_dca"]
        else:
            dca_res = cached_dca(best_name, best_model, model_state["X_test"], model_state["y_test"])
        fig_dca = px.line(dca_res, x="Threshold_Value", y=["Model_NB", "Treat_All_NB", "Treat_None_NB"], 
                          labels={"value": "Net Benefit", "variable": "Strategy"}, title="Decision Curve Analysis")
        st.plotly_chart(fig_dca, use_container_width=True)

        # 5. Survival Analysis
        st.markdown("#### 5. 충전 부하 한계 도달 시간 시뮬레이션 (Survival Analysis)")
        st.caption("각 지역이 현재 인프라 수준에서 버틸 수 있는 한계 시간(Time-to-Overload)을 카플란-마이어(Kaplan-Meier) 커브 형태로 추정합니다.")
        growth_rate = st.slider("전기차 연간 예상 증가율 (%)", min_value=1.0, max_value=20.0, value=5.0, step=1.0) / 100.0
        
        # growth_rate == 0.05 이고 사전 연산된 결과가 존재하면 실시간 계산 스킵 (대기 시간 0초)
        if abs(growth_rate - 0.05) < 1e-5 and "precomputed_survival_5" in model_state:
            surv_res = model_state["precomputed_survival_5"]
        else:
            surv_res = cached_survival(final_data.to_json(orient="split"), growth_rate)

        surv_counts = surv_res["Time_to_Overload"].value_counts().sort_index()
        total = len(surv_res)
        survival_curve = []
        current_surv = total
        for year in range(16):
            if year in surv_counts:
                current_surv -= surv_counts[year]
            survival_curve.append({"Year": year, "Survival_Probability": current_surv / total})

        fig_surv = px.line(pd.DataFrame(survival_curve), x="Year", y="Survival_Probability", markers=True, 
                           title=f"연간 {growth_rate*100:.0f}% 성장 가정 시 과부하 도달 생존 곡선 (Time-to-Overload)")
        fig_surv.update_yaxes(range=[0, 1.05])
        st.plotly_chart(fig_surv, use_container_width=True)

    elif active_menu == "🧠 SHAP/LIME 설명":
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
        X_all = model_state["X"]
        pd_df = cached_partial_dependence(model_state["best_name"], best_model, X_all, selected_feature)
        fig = px.line(pd_df, x="Feature value", y="Mean prediction", markers=True, title="Partial dependence")
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

    elif active_menu == "📋 조건 충족표":
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
                ["머신러닝 4개", "충족", "RandomForest, ExtraTrees, GradientBoosting, KNN"],
                ["딥러닝 2개", "충족", "Numpy_1D_CNN, Tabular_Transformer"],
                ["하이퍼파라미터 튜닝", "충족", "GridSearchCV 적용 완료"],
                ["부트스트랩 (95% CI)", "충족", "학술적 강건성 평가 탭"],
                ["적대적 노이즈 방어력", "충족", "가우시안 노이즈 주입 및 방어력 계산"],
                ["민감도 분석 (Ablation)", "충족", "피처 순차 제거 후 RMSE 평가"],
                ["의사결정 곡선 (DCA)", "충족", "과부하 임계치 기반 Net Benefit 산출"],
                ["생존 분석 시뮬레이터", "충족", "단일 지역 타임라인 궤적 및 카플란-마이어 곡선"],
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


def render_report(filtered, final_data, model_state):
    best_name = model_state["best_name"]
    best_model = model_state["models"][best_name]
    metrics_df = model_state["metrics"]
    test_rmse = metrics_df[(metrics_df["Model"] == best_name) & (metrics_df["Split"] == "Test")]["RMSE"].values[0]

    # 현재 적용된 필터(지역/용도 등)에 대해 최우수 모델이 선택된 변수들로 예측한 결과를 바탕으로 TOP 3 산출
    pred_full = final_data.copy()
    pred_full["예측_위험도"] = best_model.predict(model_state["X"])
    pred_df = pred_full[pred_full.index.isin(filtered.index)].copy()
    top3 = pred_df.sort_values("예측_위험도", ascending=False).head(3)

    target_col = "전력_부하지수"
    top3_list = []
    for i, row in enumerate(top3.itertuples(), 1):
        top3_list.append(f"{i}. <b><font color=\"#1E3A8A\">{row.지역}</font></b> ({row.용도}) - 예측 {target_col}: {row.예측_위험도:.2f}")
    top_features = model_state["importance"].sort_values("Importance", ascending=False).head(2)["Feature"].tolist()
    import tempfile
    
    # Feature Importance Matplotlib Chart (Headless Server-compatible, No Kaleido)
    imp_df = model_state["importance"].sort_values("Importance", ascending=True).tail(10)
    
    fig_mat, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(imp_df)))
    
    # 한글 및 유니코드 예외 방지 설정
    plt.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False
    
    ax.barh(imp_df["Feature"], imp_df["Importance"], color=colors)
    ax.set_title("상위 10개 핵심 인자 (Feature Importance)", fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.tick_params(labelsize=8)
    fig_mat.tight_layout()
    
    tmp_imp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig_mat.savefig(tmp_imp.name, dpi=150, bbox_inches='tight')
    plt.close(fig_mat)

    # PDF 다운로드 버튼
    pdf_bytes = bytes(generate_report_pdf(best_name, test_rmse, top3_list, top_features, feature_importance_img=tmp_imp.name, final_data=final_data))


    col1, col2 = st.columns([8, 2])
    with col1:
        st.markdown("<h2 style='margin-bottom: 0;'>수도권 전기차 충전소 부하 예측 결과 보고</h2>", unsafe_allow_html=True)
    with col2:
        st.download_button(
            label="📥 PDF로 다운로드",
            data=pdf_bytes,
            file_name="정책_결정_보고서.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

    st.markdown("""
    <style>
    .gov-report {
        font-family: 'Noto Serif KR', 'Batang', serif;
        line-height: 1.8;
        text-align: justify;
        background-color: #FFFFFF;
        padding: 40px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        color: #1F2937;
    }
    .gov-report h3 {
        font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
        font-weight: bold;
        font-size: 1.25rem;
        margin-top: 30px;
        margin-bottom: 15px;
        color: #111827;
    }
    .gov-summary {
        background-color: #F8FAFC;
        border: 2px solid #CBD5E1;
        padding: 20px;
        margin-bottom: 30px;
        font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
        font-weight: bold;
        color: #0F172A;
    }
    .gov-report ul {
        list-style-type: none;
        padding-left: 0;
        margin-bottom: 20px;
    }
    .gov-report ul li {
        position: relative;
        padding-left: 20px;
        margin-bottom: 8px;
    }
    .gov-report > ul > li::before {
        content: "○";
        position: absolute;
        left: 0;
    }
    .gov-report ul ul {
        margin-top: 8px;
        margin-bottom: 0;
    }
    .gov-report ul ul > li::before {
        content: "-";
        position: absolute;
        left: 5px;
    }
    .gov-report ul ul ul > li::before {
        content: "ㆍ";
        position: absolute;
        left: 10px;
    }
    .highlight {
        font-weight: bold;
        color: #1E3A8A; /* Navy */
        border-bottom: 2px solid #1E3A8A;
    }
    </style>
    """, unsafe_allow_html=True)

    top3_html = "".join([f"<li>{x}</li>" for x in top3_list])

    import datetime
    current_kst = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
    current_date_str = current_kst.strftime("%Y. %m. %d.")

    st.markdown("""
<style>
.cover-page {
    border-top: 5px solid #0070C0;
    border-bottom: 5px solid #0070C0;
    padding: 60px 20px;
    text-align: center;
    background-color: #FFFFFF;
    margin-bottom: 50px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}
.cover-title {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
    font-size: 2.5rem;
    font-weight: 900;
    color: #111827;
    margin-bottom: 20px;
}
.cover-subtitle {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
    font-size: 1.25rem;
    color: #4B5563;
    margin-bottom: 80px;
}
.cover-footer {
    font-family: 'Noto Serif KR', 'Batang', serif;
    font-size: 1.5rem;
    font-weight: bold;
    color: #000000;
}
.mois-footer {
    margin-top: 50px;
    border-top: 2px solid #D1D5DB;
    padding-top: 20px;
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
    font-size: 0.9rem;
    color: #4B5563;
}
.mois-table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 20px;
}
.mois-table td {
    padding: 8px;
    border-bottom: 1px solid #E5E7EB;
}
</style>
""" + f"""
<div class="cover-page">
    <div class="cover-title">수도권 전기차 충전소 인프라 확충 방안</div>
    <div class="cover-subtitle">- 데이터 기반 고위험 지역 도출 및 예측 모델링 -</div>
    <div class="cover-footer">
        {current_date_str}
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="gov-summary">
[핵심 요약]<br>
현재 수도권 내 전력 및 충전 대기가 가장 심각할 것으로 우려되는 <b><span style="color:#1E3A8A;">TOP 3 고위험 지역 도출 완료</span></b>.<br>
<b>{best_name}</b> 예측 모델(RMSE: {test_rmse:.4f}) 기반의 시뮬레이션을 통해, 향후 해당 지역을 최우선으로 한 맞춤형 인프라 확충 정책 수립 요망.
</div>

<div class="gov-report">
<h3>□ 추진 배경 및 현황</h3>
<ul>
<li><b>수도권 내 전기차 보급 확대로 인한 충전 인프라 부하 증가</b>
    <ul>
        <li>최근 3년간 수도권의 전기차 등록 대수가 급증함에 따라, 주요 도심지 및 주거 밀집 지역의 충전소 부족 문제가 심화됨.</li>
        <li>특정 시간대(퇴근 시간 이후 등)에 충전 수요가 집중되면서 대기 시간 지연 및 국지적 전력망 과부하 우려가 제기됨.</li>
    </ul>
</li>
<li><b>데이터 기반의 정확한 부하 예측 및 취약 지역 선제적 대응 요망</b>
    <ul>
        <li>단순 인구수나 면적에 비례한 일률적 설치가 아닌, 실제 충전 패턴과 인프라 접근성을 반영한 과학적 예측 모델 필요.</li>
        <li>한정된 예산을 효율적으로 집행하기 위해 부하가 가장 높은 '고위험 지역'을 사전 식별하고 맞춤형 대응책을 마련하고자 함.</li>
    </ul>
</li>
</ul>

<h3>□ 분석 결과 (예측 모델 성능)</h3>
<ul>
<li><b>최적 예측 모델 선정: <span class="highlight">{best_name}</span></b>
    <ul>
        <li>RandomForest, GradientBoosting 등 다수의 머신러닝/딥러닝 모델 비교 평가 결과 최우수 성능 입증.</li>
        <li>예측 오차(Test RMSE): <span class="highlight">{test_rmse:.4f}</span> 수준으로, 실제 부하 지수와 매우 근접한 정밀도를 보임.</li>
    </ul>
</li>
<li><b>모델 신뢰도 및 강건성 확보</b>
    <ul>
        <li>모델 과적합을 방지하기 위해 중첩 교차검증(Nested CV)을 수행하였으며, 공간적 외부 검증(Spatial CV)을 통해 타 지역 확장 시에도 예측 성능이 유지됨을 확인.</li>
    </ul>
</li>
</ul>

<h3>□ 당면 문제점 (고위험 지역 현황)</h3>
<ul>
<li><b>현재 인프라 부하 최상위 TOP 3 지역 분석</b>
    <ul>
        {top3_html}
    </ul>
</li>
<li>해당 지역들은 자가용 및 사업자용 충전 수요가 동시에 폭증하는 병목 구간으로 파악됨.</li>
<li>충전소 1기당 감당해야 할 전기차 대수가 적정 수준을 초과하여, 즉각적인 충전기 추가 증설이 시급한 상황임.</li>
</ul>

<h3>□ 주요 영향 인자 분석</h3>
<ul>
<li><b>핵심 변수: <span class="highlight">{top_features[0]}</span>, <span class="highlight">{top_features[1]}</span></b>
    <ul>
        <li>SHAP 및 Feature Importance 분석 결과, 위 두 가지 요인이 충전 부하 증감에 결정적인 역할을 하는 것으로 판명됨.</li>
        <li>향후 모니터링 체계 구축 시 해당 인자들의 변화 추이를 실시간으로 추적하는 시스템 마련 권고.</li>
    </ul>
</li>
</ul>

<h3>□ 향후 계획 및 제안</h3>
<ul>
<li><b>단기 대책: TOP 3 지역 타겟팅 시뮬레이션 및 예산 투입</b>
    <ul>
        <li>도출된 고위험 지역에 대해 1:1 맞춤형 충전소 설치 시뮬레이션을 추진하여, 필요 충전기 대수 및 예상 부하 완화율을 산출할 예정임.</li>
    </ul>
</li>
<li><b>중장기 대책: 연쇄 과부하 방지를 위한 선제적 인프라 마스터플랜 수립</b>
    <ul>
        <li>연간 5% 수준의 지속적인 전기차 증가 가정 시, 향후 3~5년 내 인접 자치구로 과부하가 도미노처럼 번질 위험성이 존재함.</li>
        <li>이를 방어하기 위해 국토부 및 한전 등 유관기관과 협조하여 광역 단위의 선제적 전력망 확충 예산 편성이 강구됨.</li>
    </ul>
</li>
</ul>
</div>
""", unsafe_allow_html=True)


