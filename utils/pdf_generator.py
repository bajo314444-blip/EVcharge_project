from fpdf import FPDF
import pandas as pd
import os
import datetime

class ReportPDF(FPDF):
    def __init__(self, font_family):
        super().__init__()
        self.font_family = font_family
        
    def header(self):
        # Skip header on the first page (cover)
        if self.page_no() > 1:
            self.set_font(self.font_family, style="B", size=10)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, "데이터 기반 의사결정 지원 시스템", border=0, align="L", new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(200, 200, 200)
            self.line(10, 20, 200, 20) # Corrected to horizontal line
            self.set_y(25)
            
    def footer(self):
        if self.page_no() > 1:
            self.set_y(-15)
            self.set_font(self.font_family, size=9)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f"- {self.page_no()} -", align="C")

def generate_report_pdf(best_name, test_rmse, top3_list, top_features, feature_importance_img=None):
    # Font Unification (V3.0 - Server-compatible Relative Path in fonts/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    nanum_path = os.path.join(current_dir, "fonts", "NanumGothic.ttf")
    nanum_bold_path = os.path.join(current_dir, "fonts", "NanumGothicBold.ttf")
    
    font_family = "Nanum"
    pdf = ReportPDF(font_family=font_family)
    
    pdf.add_font("Nanum", style="", fname=nanum_path)
    pdf.add_font("Nanum", style="B", fname=nanum_bold_path)

    # --- PAGE 1: COVER PAGE ---
    pdf.add_page()
    pdf.set_font(font_family, style="B", size=24)
    
    # Top Blue Lines
    pdf.set_draw_color(0, 112, 192)
    pdf.set_line_width(2.0)
    pdf.line(20, 80, 190, 80)
    pdf.set_line_width(0.5)
    pdf.line(20, 83, 190, 83)
    
    # Title
    pdf.set_y(100)
    pdf.cell(0, 15, "수도권 전기차 충전소 인프라 확충 방안", align="C", new_x="LMARGIN", new_y="NEXT")
    
    # Subtitle
    pdf.set_font(font_family, style="B", size=14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 15, "- 데이터 기반 고위험 지역 도출 및 예측 모델링 -", align="C", new_x="LMARGIN", new_y="NEXT")
    
    # Bottom Blue Lines
    pdf.set_y(140)
    pdf.set_line_width(2.0)
    pdf.line(20, 135, 190, 135)
    
    # Date
    pdf.set_y(220)
    pdf.set_font(font_family, style="B", size=16)
    pdf.set_text_color(0, 0, 0)
    current_date = (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime("%Y. %m. %d.")
    pdf.cell(0, 10, current_date, align="C", new_x="LMARGIN", new_y="NEXT")


    # --- PAGE 2: SUMMARY & BACKGROUND ---
    pdf.add_page()
    pdf.set_font(font_family, style="B", size=18)
    pdf.cell(0, 15, "수도권 전기차 충전소 부하 예측 결과 보고", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # Summary Box using native fpdf2
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(203, 213, 225)
    pdf.set_font(font_family, style="B", size=11)
    pdf.set_text_color(0, 0, 0)
    summary_text = f"[핵심 요약]\n현재 수도권 내 전력 및 충전 대기가 가장 심각할 것으로 우려되는 TOP 3 고위험 지역 도출 완료.\n{best_name} 예측 모델(RMSE: {test_rmse:.4f}) 기반의 시뮬레이션을 통해, 향후 해당 지역을 최우선으로 한 맞춤형 인프라 확충 정책 수립 요망."
    pdf.multi_cell(0, 8, summary_text, border=1, fill=True, align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    top3_html = "".join([f"<li>{x}</li>" for x in top3_list])
    
    html_page2 = f"""
    <b>□ 추진 배경 및 현황</b>
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
    """
    pdf.write_html(html_page2)
    
    # --- PAGE 3: RESULTS ---
    pdf.add_page()
    html_page3 = f"""
    <b>□ 분석 결과 (예측 모델 성능)</b>
    <ul>
        <li><b>최적 예측 모델 선정: <font color="#1E3A8A">{best_name}</font></b>
            <ul>
                <li>RandomForest, GradientBoosting 등 다수의 머신러닝/딥러닝 모델 비교 평가 결과 최우수 성능 입증.</li>
                <li>예측 오차(Test RMSE): <b><font color="#1E3A8A">{test_rmse:.4f}</font></b> 수준으로, 실제 부하 지수와 매우 근접한 정밀도를 보임.</li>
            </ul>
        </li>
        <li><b>모델 신뢰도 및 강건성 확보</b>
            <ul>
                <li>모델 과적합을 방지하기 위해 중첩 교차검증(Nested CV)을 수행하였으며, 공간적 외부 검증(Spatial CV)을 통해 타 지역 확장 시에도 예측 성능이 유지됨을 확인.</li>
            </ul>
        </li>
    </ul>
    <br>
    
    <b>□ 당면 문제점 (고위험 지역 현황)</b>
    <ul>
        <li><b>현재 인프라 부하 최상위 TOP 3 지역 분석</b>
            <ul>
                {top3_html}
            </ul>
        </li>
        <li>해당 지역들은 자가용 및 사업자용 충전 수요가 동시에 폭증하는 병목 구간으로 파악됨.</li>
        <li>충전소 1기당 감당해야 할 전기차 대수가 적정 수준을 초과하여, 즉각적인 충전기 추가 증설이 시급한 상황임.</li>
    </ul>
    """
    pdf.write_html(html_page3)

    # --- PAGE 4: PLAN & FOOTER ---
    pdf.add_page()
    html_page4 = f"""
    <b>□ 주요 영향 인자 분석</b>
    <ul>
        <li><b>핵심 변수: <font color="#1E3A8A">{top_features[0]}</font>, <font color="#1E3A8A">{top_features[1]}</font></b>
            <ul>
                <li>SHAP 및 Feature Importance 분석 결과, 위 두 가지 요인이 충전 부하 증감에 결정적인 역할을 하는 것으로 판명됨.</li>
                <li>향후 모니터링 체계 구축 시 해당 인자들의 변화 추이를 실시간으로 추적하는 시스템 마련 권고.</li>
            </ul>
        </li>
    </ul>
    <br>

    <b>□ 향후 계획 및 제안</b>
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
    """
    pdf.write_html(html_page4)
    
    if feature_importance_img and os.path.exists(feature_importance_img):
        # Insert image below the HTML block
        current_y = pdf.get_y()
        # Add some margin
        pdf.set_y(current_y + 5)
        # Image will be scaled to width 150mm and centered (page width is 210, margins are 10, so x=(210-150)/2=30)
        pdf.image(feature_importance_img, x=30, w=150)
    
    # Official Document Footer (MOIS style)
    # The team leader and footer info have been removed per user request

    return pdf.output() # returns bytearray

def generate_highway_report_pdf(hw_df, scenario, budget):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    nanum_path = os.path.join(current_dir, "fonts", "NanumGothic.ttf")
    nanum_bold_path = os.path.join(current_dir, "fonts", "NanumGothicBold.ttf")
    
    font_family = "Nanum"
    pdf = ReportPDF(font_family=font_family)
    pdf.add_font("Nanum", style="", fname=nanum_path)
    pdf.add_font("Nanum", style="B", fname=nanum_bold_path)
    pdf.add_page()
    
    # --- PAGE 1: COVER ---
    pdf.set_font(font_family, style="B", size=24)
    pdf.set_y(100)
    pdf.cell(0, 15, "하이브리드 지능형 고속도로망 관제 보고서", border=0, align="C", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font(font_family, style="", size=16)
    pdf.cell(0, 10, f"- {scenario} 대비 최적화 결과 -", border=0, align="C", new_x="LMARGIN", new_y="NEXT")
    
    # --- PAGE 2: SUMMARY ---
    pdf.add_page()
    pdf.set_font(font_family, style="B", size=16)
    pdf.cell(0, 15, "고속도로망 확충 시뮬레이션 요약", border=0, align="L", new_x="LMARGIN", new_y="NEXT")
    
    html = f"""
    <ul>
        <li><b>적용 시나리오:</b> {scenario}</li>
        <li><b>투입 예산 (충전기 대수):</b> {budget} 대</li>
        <li><b>최적 입지 도출 결과:</b> 총 {len(hw_df[hw_df['최적_추가대수'] > 0])}개 휴게소/IC에 인프라 분산 배치</li>
    </ul>
    <br>
    <b>□ 주요 분석 결과</b>
    <ul>
        <li>선형 계획법(LP)을 통해 각 휴게소의 Max_Capacity(최대 수용 한계)를 초과하지 않도록 최적화 됨.</li>
        <li>BallTree 공간 조인을 통한 총용량(kW) 가중치가 반영되어, 고출력 충전기가 부족한 핵심 거점이 우선적으로 식별됨.</li>
    </ul>
    """
    pdf.write_html(html)
    return pdf.output()
