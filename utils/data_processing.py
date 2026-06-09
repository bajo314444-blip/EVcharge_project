# ============================================================
# 파일명: data_processing.py
# 설명: 전기차 충전 인프라 데이터 로드, 전처리, 피처 엔지니어링,
#       고속도로 공간 매칭, 사전 계산 분석 결과 복원을 담당하는 모듈
# ============================================================

import numpy as np  # numpy(넘파이) 수치 연산 라이브러리를 np로 import(임포트)
import pandas as pd  # pandas(판다스) 데이터 분석 라이브러리를 pd로 import(임포트)
import streamlit as st  # streamlit(스트림릿) 웹 앱 프레임워크를 st로 import(임포트)
from pathlib import Path  # Path(경로) 클래스를 import(임포트)하여 파일 경로 처리
from sklearn.experimental import enable_iterative_imputer  # IterativeImputer(반복 대체) 실험 기능 활성화 import(임포트)
from sklearn.impute import IterativeImputer  # IterativeImputer(반복 대체) 결측치 보간 클래스 import(임포트)
import json  # json(JSON) 직렬화/역직렬화 모듈 import(임포트)
from sklearn.neighbors import BallTree  # BallTree(볼 트리) 공간 인덱스 클래스 import(임포트)

# --- 기본 데이터 디렉터리 및 원본 파일명 상수 정의 ---
DEFAULT_DATA_DIR = Path(r"./dataset")  # 기본 dataset(데이터셋) 폴더 경로 상수

SEOUL_FILE = "서울 전기차 충전소 설치현황.xlsx"  # 서울 충전소 설치현황 Excel(엑셀) 파일명
GYEONGGI_FILE = "경기도 전기차 충전소 설치현황.xlsx"  # 경기도 충전소 설치현황 Excel(엑셀) 파일명
SALES_FILE = "전국 전기차 충전 전력 판매실적.csv"  # 전력 판매실적 CSV 파일명
REGISTER_FILE = "한국교통안전공단_전국_전기차_차종별_용도별_차량_등록대수(운행차량기준)_20250407.csv"  # 전기차 등록대수 CSV 파일명
HOURLY_LOAD_FILE = "한국전력공사_전기차 시간대별 충전부하_20240930.csv"  # 시간대별 충전부하 CSV 파일명
LOCATION_FILE = "한국전력공사_전기차충전소위경도_20250531.csv"  # 충전소 위·경도 CSV 파일명
CAPACITY_FILE = "한국전력공사_충전소별 충전기 용량 정보_20240603.csv"  # 충전기 용량 정보 CSV 파일명
MONTHLY_PUBLIC_FAST_FILE = (  # 환경부 공공급속 월별 충전량 CSV 파일명 (긴 문자열 분할)
    "환경부 공공급속 충전기 연월별 충전량, 충전횟수, 충전시간(2015년 ~ 2025년8월)_"
    "년월별_20250923_FF.csv"
)

# --- 수도권 시·도명 매핑 상수 ---
METRO_LONG = ["서울특별시", "경기도", "인천광역시"]  # 정식 시·도명 목록 (필터링용)
METRO_SHORT = ["서울", "경기", "인천"]  # 축약 시·도명 목록 (집계·병합 키용)
SIDO_MAP = {  # 시·도명 정규화(normalize) 매핑 dictionary(딕셔너리)
    "서울특별시": "서울",  # 정식명 → 축약명
    "서울": "서울",  # 이미 축약된 경우 그대로 유지
    "경기도": "경기",  # 정식명 → 축약명
    "경기": "경기",  # 이미 축약된 경우 그대로 유지
    "인천광역시": "인천",  # 정식명 → 축약명
    "인천": "인천",  # 이미 축약된 경우 그대로 유지
}

# --- 시·도명 정규화 유틸리티 ---
def normalize_sido(value):
    """시·도명 문자열을 축약형(서울/경기/인천)으로 정규화하여 반환한다."""
    if pd.isna(value):  # value(값)가 결측(NaN)이면
        return ""  # 빈 문자열 반환
    return SIDO_MAP.get(str(value).strip(), str(value).strip())  # SIDO_MAP에서 조회, 없으면 공백 제거 후 원문 반환

# --- 인코딩 자동 감지 CSV 읽기 ---
def read_csv_safely(path, **kwargs):
    """여러 인코딩(utf-8-sig, cp949, euc-kr)을 순차 시도하여 CSV를 안전하게 읽는다."""
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):  # 한국어 CSV에서 흔한 encoding(인코딩) 목록 순회
        try:  # 현재 encoding(인코딩)으로 읽기 시도
            return pd.read_csv(path, encoding=encoding, **kwargs)  # 성공 시 DataFrame(데이터프레임) 반환
        except UnicodeDecodeError:  # 디코딩 실패 시
            continue  # 다음 encoding(인코딩)으로 재시도
    return pd.read_csv(path, **kwargs)  # 모든 encoding(인코딩) 실패 시 기본 설정으로 최종 시도

# --- 주소 문자열에서 시·군·구 추출 ---
def get_city_from_address(value):
    """주소 문자열에서 두 번째 토큰(시·군·구)을 추출하여 반환한다."""
    parts = str(value).strip().split()  # 주소를 공백 기준으로 split(분할)
    return parts[1] if len(parts) > 1 else ""  # 두 번째 토큰이 있으면 반환, 없으면 빈 문자열

def get_sido_from_address(value):
    """주소 문자열의 첫 번째 토큰(시·도)을 정규화하여 반환한다."""
    parts = str(value).strip().split()  # 주소를 공백 기준으로 split(분할)
    return normalize_sido(parts[0] if parts else "")  # 첫 토큰을 normalize_sido(시·도 정규화) 후 반환

# --- 전체 데이터 로드 및 전처리 (Streamlit 캐시 적용) ---
@st.cache_data(show_spinner="데이터를 불러오고 전처리하는 중입니다...")  # cache_data(데이터 캐시) 데코레이터로 재로딩 방지
def load_all_data(data_dir_text):
    """지정 경로에서 수도권 충전·수요·공급 데이터를 로드하고 통합 전처리하여 반환한다."""
    data_dir = Path(data_dir_text)  # 문자열 경로를 Path(경로) 객체로 변환
    required = [  # 필수 원본 파일명 목록
        SEOUL_FILE,  # 서울 설치현황
        GYEONGGI_FILE,  # 경기 설치현황
        SALES_FILE,  # 전력 판매실적
        REGISTER_FILE,  # 전기차 등록대수
        LOCATION_FILE,  # 충전소 위·경도
        CAPACITY_FILE,  # 충전기 용량
        MONTHLY_PUBLIC_FAST_FILE,  # 월별 공공급속 충전량
    ]
    missing = [name for name in required if not (data_dir / name).exists()]  # 존재하지 않는 필수 파일 목록 수집
    if missing:  # 누락 파일이 있으면
        raise FileNotFoundError("데이터 폴더에서 찾지 못한 파일: " + ", ".join(missing))  # FileNotFoundError(파일 없음 예외) 발생

    # --- 원본 파일 로드 ---
    seoul_built = pd.read_excel(data_dir / SEOUL_FILE)  # 서울 충전소 설치현황 Excel(엑셀) 로드
    gyeonggi_built = pd.read_excel(data_dir / GYEONGGI_FILE)  # 경기 충전소 설치현황 Excel(엑셀) 로드
    sales = read_csv_safely(data_dir / SALES_FILE)  # 전력 판매실적 CSV 안전 로드
    register = read_csv_safely(data_dir / REGISTER_FILE)  # 전기차 등록대수 CSV 안전 로드
    location = read_csv_safely(data_dir / LOCATION_FILE)  # 충전소 위·경도 CSV 안전 로드
    capacity = read_csv_safely(data_dir / CAPACITY_FILE)  # 충전기 용량 CSV 안전 로드

    # --- 수요(전력 판매) 데이터 전처리 ---
    sales["시도"] = sales["시도"].map(normalize_sido)  # 시·도명을 축약형으로 정규화
    sales = sales[sales["시도"].isin(METRO_SHORT)].copy()  # 수도권(서울/경기/인천)만 필터링
    sales["용도"] = sales["충전요금"].replace({"사업용": "사업자용"})  # 용도 컬럼명 통일 (사업용 → 사업자용)
    sales["시군구"] = sales["시군구"].astype(str).str.split().str[0]  # 시·군·구 문자열에서 첫 토큰만 추출
    sales["판매량"] = pd.to_numeric(sales["판매량"], errors="coerce").fillna(0)  # 판매량을 numeric(숫자형)으로 변환, 결측은 0
    sales["판매수입"] = pd.to_numeric(sales["판매수입"], errors="coerce").fillna(0)  # 판매수입을 numeric(숫자형)으로 변환, 결측은 0
    demand_sales = (  # 시·도·시군구·용도별 전력 판매 집계
        sales.groupby(["시도", "시군구", "용도"], as_index=False)  # groupby(그룹화) 키 설정
        .agg(총_전력판매량=("판매량", "sum"), 총_판매수입=("판매수입", "sum"))  # agg(집계)로 합계 계산
    )

    # --- 수요(전기차 등록) 데이터 전처리 ---
    register = register[register["시군구별"].astype(str).str.contains("서울|경기|인천", na=False)].copy()  # 수도권 행만 필터링
    register["시도"] = register["시군구별"].apply(lambda x: normalize_sido(str(x).split()[0]))  # 시·도명 정규화
    register["시군구"] = register["시군구별"].apply(get_city_from_address)  # 주소에서 시·군·구 추출
    register["용도"] = register["용도별"].replace({"비사업용": "자가용", "사업용": "사업자용"})  # 용도 라벨 통일
    register["계"] = pd.to_numeric(register["계"], errors="coerce").fillna(0)  # 등록대수를 numeric(숫자형)으로 변환
    demand_ev = (  # 시·도·시군구·용도별 전기차 등록대수 집계
        register.groupby(["시도", "시군구", "용도"], as_index=False)  # groupby(그룹화) 키 설정
        .agg(전기차_전체대수=("계", "sum"))  # agg(집계)로 등록대수 합계
    )
    demand = pd.merge(demand_ev, demand_sales, on=["시도", "시군구", "용도"], how="outer").fillna(0)  # 수요 DataFrame(데이터프레임) outer merge(병합) 후 결측 0

    # --- 설치현황(공급) 데이터 전처리 ---
    built = pd.concat([seoul_built, gyeonggi_built], ignore_index=True)  # 서울·경기 설치현황 concat(결합)
    built["시도"] = built["주소"].apply(get_sido_from_address)  # 주소에서 시·도 추출
    built["시군구"] = built["주소"].apply(get_city_from_address)  # 주소에서 시·군·구 추출
    for col in ["급속충전기(대)", "완속충전기(대)"]:  # 충전기 대수 컬럼 순회
        built[col] = pd.to_numeric(built[col], errors="coerce").fillna(0)  # numeric(숫자형) 변환, 결측 0
    built_summary = (  # 시·도·시군구별 설치 충전기 대수 집계
        built.groupby(["시도", "시군구"], as_index=False)  # groupby(그룹화) 키 설정
        .agg(급속충전기_대수=("급속충전기(대)", "sum"), 완속충전기_대수=("완속충전기(대)", "sum"))  # 급속/완속 대수 합계
    )

    # --- 위치·용량 기반 공급 데이터 전처리 ---
    location = location[location["충전소주소"].astype(str).str.contains("|".join(METRO_LONG), na=False)].copy()  # 수도권 충전소만 필터링
    location["시도"] = location["충전소주소"].apply(get_sido_from_address)  # 주소에서 시·도 추출
    location["시군구"] = location["충전소주소"].apply(get_city_from_address)  # 주소에서 시·군·구 추출
    capacity["충전기용량(kw)"] = pd.to_numeric(capacity["충전기용량(kw)"], errors="coerce").fillna(0)  # 충전기 용량 numeric(숫자형) 변환

    # --- 용량·위치 DataFrame(데이터프레임) merge(병합) ---
    if "충전소ID" in capacity.columns and "충전소ID" in location.columns:  # 충전소ID 컬럼이 양쪽 모두 있으면
        supply = pd.merge(  # 충전소ID 기준 right merge(병합)
            capacity,  # 용량 DataFrame(데이터프레임)
            location[["충전소ID", "충전소명", "충전소주소", "시도", "시군구", "위도", "경도"]],  # 위치 관련 컬럼만 선택
            on="충전소ID",  # merge(병합) 키
            how="right",  # location(위치) 기준 right join(조인)
            suffixes=("_용량", ""),  # 중복 컬럼 suffix(접미사) 설정
        )
        supply["충전소명"] = supply["충전소명"].fillna(supply.get("충전소명_용량", ""))  # 충전소명 결측 시 용량 쪽 이름으로 보완
    else:  # 충전소ID가 없으면
        supply = pd.merge(  # 충전소명 기준 right merge(병합)
            capacity,  # 용량 DataFrame(데이터프레임)
            location[["충전소명", "충전소주소", "시도", "시군구", "위도", "경도"]],  # 위치 관련 컬럼만 선택
            on="충전소명",  # merge(병합) 키
            how="right",  # location(위치) 기준 right join(조인)
        )

    supply = supply.dropna(subset=["위도", "경도"]).copy()  # 위·경도 결측 행 제거
    supply_summary = (  # 시·도·시군구별 공급 인프라 집계
        supply.groupby(["시도", "시군구"], as_index=False)  # groupby(그룹화) 키 설정
        .agg(  # agg(집계) 함수 적용
            충전소개수=("충전소명", "nunique"),  # 고유 충전소 수
            충전기대수=("충전기용량(kw)", "count"),  # 충전기 행 수
            총용량_kW=("충전기용량(kw)", "sum"),  # 총 용량 합계
            위도=("위도", "mean"),  # 지역 중심 위도 평균
            경도=("경도", "mean"),  # 지역 중심 경도 평균
        )
    )

    # --- 수요·공급 통합 및 결측치 보간 ---
    supply_total = pd.merge(built_summary, supply_summary, on=["시도", "시군구"], how="outer")  # 설치현황·용량 공급 outer merge(병합)
    final = pd.merge(demand, supply_total, on=["시도", "시군구"], how="inner")  # 수요·공급 inner merge(병합)로 최종 DataFrame(데이터프레임) 생성

    num_cols = ["전기차_전체대수", "총_전력판매량", "총_판매수입", "급속충전기_대수", "완속충전기_대수", "충전소개수", "충전기대수", "총용량_kW"]  # imputation(대체) 대상 numeric(숫자형) 컬럼 목록
    # merge(병합)로 생긴 결측치를 IterativeImputer(반복 대체)로 보간
    imputer = IterativeImputer(random_state=42, max_iter=10)  # IterativeImputer(반복 대체) 객체 생성 (난수 시드 42, 최대 10회 반복)
    final[num_cols] = imputer.fit_transform(final[num_cols])  # fit_transform(학습·변환)으로 결측치 대체
    # imputation(대체) 후 음수 값이 생기지 않도록 하한 0으로 clip(클리핑)
    final[num_cols] = final[num_cols].clip(lower=0)  # lower=0으로 음수 제거

    # --- 파생 변수(부하 지수 등) 생성 ---
    final["전체_충전기대수"] = final["급속충전기_대수"] + final["완속충전기_대수"]  # 급속+완속 충전기 총 대수
    final["전체_충전기대수"] = final["전체_충전기대수"].where(final["전체_충전기대수"] > 0, final["충전기대수"])  # 0이면 용량 기반 대수로 대체
    final = final[(final["전체_충전기대수"] > 0) & (final["총용량_kW"] > 0)].copy()  # 충전기·용량이 0인 행 제외
    final["인프라_부하지수"] = final["전기차_전체대수"] / final["전체_충전기대수"]  # 전기차 대수 대비 충전기 부하 지수
    final["전력_부하지수"] = final["총_전력판매량"] / final["총용량_kW"]  # 전력 판매량 대비 용량 부하 지수
    final["전력_부하지수"] = final["전력_부하지수"].replace([np.inf, -np.inf], np.nan).fillna(0)  # inf(무한대)를 NaN 후 0으로 대체
    final["인프라_부하지수"] = final["인프라_부하지수"].replace([np.inf, -np.inf], np.nan).fillna(0)  # inf(무한대)를 NaN 후 0으로 대체
    final["지역"] = final["시도"] + " " + final["시군구"]  # 시·도+시군구 결합 지역명 컬럼 생성

    # --- 2.5단계: 피처 엔지니어링 (다중공선성 제거 및 파생 변수 생성) ---
    from sklearn.decomposition import PCA  # PCA(주성분 분석) 클래스 import(임포트)
    from sklearn.preprocessing import StandardScaler, MinMaxScaler  # StandardScaler(표준화), MinMaxScaler(최소-최대 스케일링) import(임포트)

    infra_cols = ["급속충전기_대수", "완속충전기_대수", "총용량_kW"]  # PCA(주성분 분석) 입력 인프라 컬럼
    pca = PCA(n_components=1, random_state=42)  # 1차원 PCA(주성분 분석) 객체 생성
    infra_scaled = StandardScaler().fit_transform(final[infra_cols])  # StandardScaler(표준화) 후 transform(변환)
    pca_raw = pca.fit_transform(infra_scaled)[:, 0]  # PCA(주성분 분석) 1차 성분 추출
    # PCA 특성상 평균이 0이 되고 음수가 발생하므로, 직관적인 해석을 위해 0~100점 척도의 Index(지수)로 변환
    final["충전인프라_규모_PCA"] = MinMaxScaler(feature_range=(0, 100)).fit_transform(pca_raw.reshape(-1, 1)).flatten()  # MinMaxScaler(0~100) 적용

    final["충전기_1대당_평균용량"] = final["총용량_kW"] / final["전체_충전기대수"]  # 충전기 1대당 평균 kW 계산
    final["충전기_1대당_평균용량"] = final["충전기_1대당_평균용량"].replace([np.inf, -np.inf], np.nan).fillna(0)  # inf(무한대) 처리

    # --- 월별 공공급속 충전량 데이터 전처리 ---
    monthly_cols = [  # 월별 CSV에서 읽을 컬럼 목록
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
    monthly = read_csv_safely(data_dir / MONTHLY_PUBLIC_FAST_FILE, usecols=monthly_cols)  # 필요 컬럼만 선택하여 CSV 로드
    monthly = monthly[monthly["시도"].isin(METRO_LONG)].copy()  # 수도권 시·도만 필터링
    monthly["시도"] = monthly["시도"].map(normalize_sido)  # 시·도명 축약형 정규화
    monthly = monthly.rename(columns={"군구": "시군구"})  # 컬럼명 통일 (군구 → 시군구)
    monthly["연월"] = pd.to_datetime(  # 연·월을 datetime(날짜시간) 연월 컬럼으로 변환
        monthly["연"].astype(str) + "-" + monthly["월"].astype(str).str.zfill(2) + "-01"  # YYYY-MM-01 형식 문자열 생성
    )
    for col in ["충전량_합산", "충전횟수_합산", "충전시간_합산"]:  # 월별 집계 numeric(숫자형) 컬럼 순회
        monthly[col] = pd.to_numeric(monthly[col], errors="coerce").fillna(0)  # numeric(숫자형) 변환, 결측 0

    monthly_region = (  # 연월·시·도·시군구별 월별 충전 집계
        monthly.groupby(["연월", "시도", "시군구"], as_index=False)  # groupby(그룹화) 키 설정
        .agg(  # agg(집계) 함수 적용
            월_충전량=("충전량_합산", "sum"),  # 월 충전량 합계
            월_충전횟수=("충전횟수_합산", "sum"),  # 월 충전횟수 합계
            월_충전시간=("충전시간_합산", "sum"),  # 월 충전시간 합계
            운영_충전소수=("충전소ID", "nunique"),  # 운영 충전소 고유 개수
        )
    )
    monthly_region = pd.merge(  # 월별 집계에 지역별 총용량·좌표 merge(병합)
        monthly_region,  # 월별 지역 집계 DataFrame(데이터프레임)
        supply_total[["시도", "시군구", "총용량_kW", "위도", "경도"]],  # 공급 요약에서 필요 컬럼만 선택
        on=["시도", "시군구"],  # merge(병합) 키
        how="left",  # left join(조인)으로 월별 데이터 기준 유지
    )
    monthly_region["월별_부하지수"] = monthly_region["월_충전량"] / monthly_region["총용량_kW"].replace(0, np.nan)  # 월 충전량/총용량 부하 지수
    monthly_region["월별_부하지수"] = monthly_region["월별_부하지수"].replace([np.inf, -np.inf], np.nan)  # inf(무한대)를 NaN으로 대체
    monthly_region["지역"] = monthly_region["시도"] + " " + monthly_region["시군구"]  # 지역명 컬럼 생성

    # --- 시간대별 부하 데이터 (선택적 로드) ---
    hourly = read_csv_safely(data_dir / HOURLY_LOAD_FILE) if (data_dir / HOURLY_LOAD_FILE).exists() else pd.DataFrame()  # 파일 있으면 로드, 없으면 빈 DataFrame(데이터프레임)
    return final, monthly_region, hourly  # 최종 통합·월별·시간별 DataFrame(데이터프레임) 튜플 반환


# --- 고속도로 휴게소·IC 데이터 로드 및 주변 충전기 매칭 ---
def load_highway_data(data_dir_text):
    """고속도로 JSON 노드와 주변 충전소를 BallTree(볼 트리)로 매칭하여 DataFrame(데이터프레임)을 반환한다."""
    data_dir = Path(data_dir_text)  # 문자열 경로를 Path(경로) 객체로 변환
    hw_dir = data_dir / "highway_adress"  # 고속도로 주소 JSON 폴더 경로
    if not hw_dir.exists():  # 폴더가 없으면
        return pd.DataFrame()  # 빈 DataFrame(데이터프레임) 반환

    # --- JSON 파일에서 고속도로 노드 좌표 수집 ---
    highway_nodes = []  # 파싱된 고속도로 노드 dict(딕셔너리) 목록
    for fpath in hw_dir.glob("etc_page*.json"):  # etc_page*.json 패턴 파일 순회
        try:  # JSON 파싱 중 예외 무시
            with open(fpath, "r", encoding="utf-8") as f:  # UTF-8 encoding(인코딩)으로 파일 열기
                data = json.load(f)  # JSON 파싱
                for item in data.get("list", []):  # list(목록) 항목 순회
                    if item.get("xValue") and item.get("yValue"):  # 경·위도 값이 모두 있으면
                        highway_nodes.append({  # 노드 정보 dict(딕셔너리) 추가
                            "unitName": str(item["unitName"]).strip(),  # 휴게소/IC 명칭
                            "routeName": str(item.get("routeName", "알수없음")).strip(),  # 노선명 (없으면 '알수없음')
                            "경도": float(item["xValue"]),  # xValue → 경도
                            "위도": float(item["yValue"])  # yValue → 위도
                        })
        except Exception:  # 파싱·IO 예외 발생 시
            pass  # 해당 파일 건너뛰기

    if not highway_nodes:  # 수집된 노드가 없으면
        return pd.DataFrame()  # 빈 DataFrame(데이터프레임) 반환

    hw_df = pd.DataFrame(highway_nodes).dropna(subset=["위도", "경도"])  # DataFrame(데이터프레임) 생성 후 좌표 결측 제거
    # 상·하행 명칭 중복 시 좌표 기준으로 구분하기 위해 중복 제거
    hw_df = hw_df.drop_duplicates(subset=["위도", "경도"])  # 동일 좌표 중복 행 제거

    # --- 충전소 위치·용량 데이터 로드 ---
    location = read_csv_safely(data_dir / LOCATION_FILE)  # 충전소 위·경도 CSV 로드
    capacity = read_csv_safely(data_dir / CAPACITY_FILE)  # 충전기 용량 CSV 로드

    location["위도"] = pd.to_numeric(location["위도"], errors="coerce")  # 위도 numeric(숫자형) 변환
    location["경도"] = pd.to_numeric(location["경도"], errors="coerce")  # 경도 numeric(숫자형) 변환
    capacity["충전기용량(kw)"] = pd.to_numeric(capacity["충전기용량(kw)"], errors="coerce").fillna(0)  # 용량 numeric(숫자형) 변환

    # --- 용량·위치 merge(병합) ---
    if "충전소ID" in capacity.columns and "충전소ID" in location.columns:  # 충전소ID 컬럼이 양쪽 모두 있으면
        supply = pd.merge(capacity, location[["충전소ID", "충전소명", "위도", "경도"]], on="충전소ID", how="inner")  # ID 기준 inner merge(병합)
    else:  # 충전소ID가 없으면
        supply = pd.merge(capacity, location[["충전소명", "위도", "경도"]], on="충전소명", how="inner")  # 명칭 기준 inner merge(병합)

    supply = supply.dropna(subset=["위도", "경도"]).copy()  # 좌표 결측 행 제거

    # --- BallTree(볼 트리) 공간 조인: 고속도로 노드 주변 충전기 매칭 ---
    hw_coords = np.radians(hw_df[["위도", "경도"]].values)  # 고속도로 좌표를 radian(라디안)으로 변환
    supply_coords = np.radians(supply[["위도", "경도"]].values)  # 충전소 좌표를 radian(라디안)으로 변환

    tree = BallTree(supply_coords, metric='haversine')  # haversine(하버사인) 거리 metric(메트릭) BallTree(볼 트리) 생성

    # 반경 3km (3 / 6371 radian(라디안)) 내 충전소 query(조회)
    radius_km = 3.0  # 검색 반경(km)
    radius_rad = radius_km / 6371.0  # 지구 반경 6371km 기준 radian(라디안) 변환

    indices, _ = tree.query_radius(hw_coords, r=radius_rad, return_distance=True)  # query_radius(반경 조회)로 인덱스·거리 반환

    hw_capacity = []  # 노드별 매칭 충전기 총용량 목록
    hw_chargers = []  # 노드별 매칭 충전기 대수 목록

    for idx_list in indices:  # 각 고속도로 노드의 매칭 충전소 인덱스 목록 순회
        if len(idx_list) > 0:  # 반경 내 충전소가 있으면
            matched_supply = supply.iloc[idx_list]  # 매칭된 충전소 행 추출
            hw_capacity.append(matched_supply["충전기용량(kw)"].sum())  # 총용량 합계 추가
            hw_chargers.append(len(matched_supply))  # 충전기 대수 추가
        else:  # 반경 내 충전소가 없으면
            hw_capacity.append(0.0)  # 용량 0
            hw_chargers.append(0)  # 대수 0

    hw_df["총용량_kW"] = hw_capacity  # 노드별 주변 충전기 총용량 컬럼 할당
    hw_df["충전기대수"] = hw_chargers  # 노드별 주변 충전기 대수 컬럼 할당

    # Max Capacity(최대 용량) 제약: 휴게소당 최소 5대, 최대 20대 가정, 기존 대수의 2배로 clip(클리핑)
    hw_df["Max_Capacity"] = np.clip(hw_df["충전기대수"] * 2, 5, 20)  # np.clip(클리핑)으로 5~20 범위 제한

    return hw_df  # 고속도로 노드·용량 DataFrame(데이터프레임) 반환

# --- 시간대 스파이크(Time Spike) 피처 생성 ---
def create_time_spike_features(dates):
    """
    datetime(날짜시간) 리스트를 받아 Time Spike(시간 스파이크) 4대 변수를 생성하여 DataFrame(데이터프레임)으로 반환한다.

    생성 피처: is_commute_time(출퇴근 시간), is_weekend(주말),
               is_holiday(공휴일), is_golden_week(연휴/징검다리)
    """
    df = pd.DataFrame({"datetime": pd.to_datetime(dates)})  # datetime(날짜시간) Series(시리즈)로 DataFrame(데이터프레임) 생성

    # --- 1. is_commute_time(출퇴근 시간) 피처 ---
    is_weekday = df["datetime"].dt.dayofweek < 5  # dayofweek(요일) 0~4 = 평일
    is_morning_commute = df["datetime"].dt.hour.between(7, 8)  # 7:00 ~ 8:59 아침 출퇴근
    is_evening_commute = df["datetime"].dt.hour.between(17, 18)  # 17:00 ~ 18:59 저녁 출퇴근
    df["is_commute_time"] = (is_weekday & (is_morning_commute | is_evening_commute)).astype(int)  # 평일 출퇴근 시간이면 1

    # --- 4. is_weekend(주말) 피처 ---
    df["is_weekend"] = (~is_weekday).astype(int)  # 평일이 아니면 주말(1)

    # --- 2. is_holiday(공휴일) & 3. is_golden_week(연휴/징검다리) 피처 ---
    # 실제로는 pytimekr 등을 쓰지만, 프로토타입을 위해 하드코딩된 주요 공휴일 매핑
    holidays_2024 = [  # 2024년 공휴일 날짜 목록
        "2024-01-01", "2024-02-09", "2024-02-10", "2024-02-11", "2024-02-12",
        "2024-03-01", "2024-04-10", "2024-05-05", "2024-05-06", "2024-05-15",
        "2024-06-06", "2024-08-15", "2024-09-16", "2024-09-17", "2024-09-18",
        "2024-10-03", "2024-10-09", "2024-12-25"
    ]
    golden_weeks = [  # 연휴·징검다리(골든위크) 날짜 목록
        # 추석 연휴 (주말 포함)
        "2024-09-14", "2024-09-15", "2024-09-16", "2024-09-17", "2024-09-18",
        # 5월 징검다리
        "2024-05-04", "2024-05-05", "2024-05-06"
    ]

    date_str = df["datetime"].dt.strftime("%Y-%m-%d")  # datetime(날짜시간)을 YYYY-MM-DD 문자열로 변환
    df["is_holiday"] = date_str.isin(holidays_2024).astype(int)  # 공휴일이면 1
    df["is_golden_week"] = date_str.isin(golden_weeks).astype(int)  # 골든위크면 1

    return df  # Time Spike(시간 스파이크) 피처 DataFrame(데이터프레임) 반환


# --- 모델 평가 캐시 래퍼 함수들 (Streamlit cache_data 적용) ---
@st.cache_data(show_spinner="부트스트랩 CI 산출 중...")  # bootstrap(부트스트랩) CI(신뢰구간) 캐시
def cached_bootstrap(best_name, _model, X_test, y_test):
    """부트스트랩(Bootstrap) 반복으로 RMSE·R² 신뢰구간(CI)을 계산한다."""
    from utils.optimization import calculate_bootstrap_ci  # optimization 모듈에서 bootstrap CI 함수 import(임포트)
    return calculate_bootstrap_ci(_model, X_test, y_test, n_iterations=100)  # 100회 반복 bootstrap(부트스트랩) CI 반환


@st.cache_data(show_spinner="적대적 공격 방어력 평가 중...")  # adversarial(적대적) 공격 평가 캐시
def cached_adversarial(best_name, _model, X_test, y_test):
    """적대적(Adversarial) perturbation(교란) 공격에 대한 모델 방어력을 평가한다."""
    from utils.optimization import run_adversarial_attack  # optimization 모듈에서 adversarial 함수 import(임포트)
    return run_adversarial_attack(_model, X_test, y_test)  # adversarial(적대적) 공격 결과 반환


@st.cache_data(show_spinner="피처 민감도 분석 중...")  # ablation(제거) study(연구) 캐시
def cached_ablation(best_name, _model, X_train, y_train, X_test, y_test, importances):
    """피처 중요도 기반 ablation(제거) study(연구)로 민감도를 분석한다."""
    from utils.optimization import run_ablation_study  # optimization 모듈에서 ablation 함수 import(임포트)
    return run_ablation_study(_model, X_train, y_train, X_test, y_test, importances)  # ablation(제거) 결과 반환


@st.cache_data(show_spinner="DCA 곡선 산출 중...")  # DCA(결정 곡선 분석) 캐시
def cached_dca(best_name, _model, X_test, y_test):
    """Decision Curve Analysis(결정 곡선 분석, DCA) 곡선 데이터를 산출한다."""
    from utils.optimization import calculate_dca  # optimization 모듈에서 DCA 함수 import(임포트)
    return calculate_dca(_model, X_test, y_test)  # DCA(결정 곡선 분석) 결과 반환


@st.cache_data(show_spinner="중첩 10-겹 교차검증 (Nested 10-fold CV) 수행 중... (시간이 걸릴 수 있습니다)")  # nested CV(중첩 교차검증) 캐시
def cached_nested_cv(best_name, _model, X, y):
    """Nested 10-fold CV(중첩 10-겹 교차검증)로 모델 일반화 성능을 평가한다."""
    from utils.optimization import run_nested_cv  # optimization 모듈에서 nested CV 함수 import(임포트)
    return run_nested_cv(_model, X, y)  # nested CV(중첩 교차검증) 결과 반환


@st.cache_data(show_spinner="공간적 외부 검증(Spatial External Validation) 수행 중...")  # spatial(공간) external validation(외부 검증) 캐시
def cached_spatial_external_validation(best_name, _model, X, y, holdout_region):
    """특정 지역을 holdout(홀드아웃)하여 공간적 외부 검증(Spatial External Validation)을 수행한다."""
    from utils.optimization import run_spatial_external_validation  # optimization 모듈에서 spatial validation 함수 import(임포트)
    return run_spatial_external_validation(_model, X, y, holdout_region)  # spatial(공간) 검증 결과 반환


@st.cache_data(show_spinner="생존 분석 시뮬레이션 중...")  # survival(생존) analysis(분석) 캐시
def cached_survival(final_json, growth_rate):
    """전기차 성장률 가정 하에 survival(생존) analysis(분석) 시뮬레이션을 실행한다."""
    from utils.optimization import run_survival_simulation  # optimization 모듈에서 survival 함수 import(임포트)
    from io import StringIO  # StringIO(문자열 입출력) import(임포트)
    final_df = pd.read_json(StringIO(final_json), orient="split")  # JSON 문자열을 DataFrame(데이터프레임)으로 복원
    return run_survival_simulation(final_df, growth_rate)  # survival(생존) 시뮬레이션 결과 반환


@st.cache_data(show_spinner="Partial Dependence 계산 중...")  # partial dependence(부분 의존) 캐시
def cached_partial_dependence(best_name, _model, X_all, selected_feature):
    """선택 feature(특성)에 대한 Partial Dependence(부분 의존) 곡선 데이터를 계산한다."""
    grid = np.linspace(X_all[selected_feature].quantile(0.05), X_all[selected_feature].quantile(0.95), 30)  # 5~95 percentile(백분위) 구간 30점 grid(그리드)
    pd_rows = []  # partial dependence(부분 의존) 결과 행 목록
    for value in grid:  # grid(그리드) 값 순회
        X_tmp = X_all.copy()  # feature(특성) DataFrame(데이터프레임) 복사
        X_tmp[selected_feature] = value  # 선택 feature(특성)를 grid(그리드) 값으로 고정
        pd_rows.append({"Feature value": value, "Mean prediction": _model.predict(X_tmp).mean()})  # 평균 prediction(예측)값 기록
    return pd.DataFrame(pd_rows)  # Partial Dependence(부분 의존) DataFrame(데이터프레임) 반환


# --- ONNX(온엑스) 모델 래퍼 클래스 ---
class ONNXModelWrapper:
    """ONNX(온엑스) Runtime(런타임) InferenceSession(추론 세션)으로 predict(예측)를 수행하는 래퍼."""

    def __init__(self, onnx_path):
        """ONNX(온엑스) 모델 파일 경로로 InferenceSession(추론 세션)을 초기화한다."""
        import onnxruntime as ort  # onnxruntime(온엑스런타임)을 ort로 import(임포트)
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])  # CPU provider(제공자)로 세션 생성
        self.input_name = self.session.get_inputs()[0].name  # model(모델) input(입력) tensor(텐서) 이름
        self.output_name = self.session.get_outputs()[0].name  # model(모델) output(출력) tensor(텐서) 이름

    def predict(self, X):
        """입력 X에 대해 ONNX(온엑스) model(모델) predict(예측) 결과를 반환한다."""
        import numpy as np  # numpy(넘파이) import(임포트) (메서드 스코프)
        import pandas as pd  # pandas(판다스) import(임포트) (메서드 스코프)
        if isinstance(X, pd.DataFrame):  # X가 DataFrame(데이터프레임)이면
            X_arr = X.values.astype(np.float32)  # values를 float32 array(배열)로 변환
        else:  # ndarray(배열) 등이면
            X_arr = np.asarray(X, dtype=np.float32)  # float32 array(배열)로 변환

        if len(X_arr.shape) == 1:  # 1차원 입력이면
            X_arr = X_arr.reshape(1, -1)  # (1, n_features) shape(형상)으로 reshape(변형)

        preds = self.session.run([self.output_name], {self.input_name: X_arr})[0]  # InferenceSession(추론 세션) run(실행)으로 predict(예측)
        if len(preds.shape) > 1 and preds.shape[1] == 1:  # (n, 1) shape(형상)이면
            preds = preds.squeeze(axis=1)  # 1차원으로 squeeze(압축)
        return preds  # predict(예측) 결과 array(배열) 반환


# --- 사전 계산 분석 결과(JSON + ONNX) 로드 ---
def load_precomputed_analytics(json_path, onnx_path):
    """사전 계산된 analytics(분석) JSON과 ONNX(온엑스) model(모델)을 로드하여 model_state(모델 상태) dict(딕셔너리)를 반환한다."""
    import json  # json(JSON) 모듈 import(임포트) (함수 스코프)
    import os  # os(운영체제) 모듈 import(임포트)
    import pandas as pd  # pandas(판다스) import(임포트) (함수 스코프)
    import numpy as np  # numpy(넘파이) import(임포트) (함수 스코프)

    if not os.path.exists(json_path) or not os.path.exists(onnx_path):  # JSON 또는 ONNX 파일이 없으면
        raise FileNotFoundError("ONNX model or JSON precomputed analytics file not found.")  # FileNotFoundError(파일 없음 예외) 발생

    with open(json_path, 'r', encoding='utf-8') as f:  # JSON 파일 UTF-8로 열기
        data = json.load(f)  # JSON 파싱

    # --- 기본 metrics(지표)·importance(중요도)·predictions(예측) DataFrame(데이터프레임) 복원 ---
    metrics = pd.DataFrame(data["metrics"])  # 평가 metrics(지표) DataFrame(데이터프레임)
    importance = pd.DataFrame(data["importance"])  # feature importance(특성 중요도) DataFrame(데이터프레임)
    predictions = pd.DataFrame(data["predictions"])  # predictions(예측) 결과 DataFrame(데이터프레임)

    X = pd.DataFrame(data["X"])  # 전체 feature(특성) X DataFrame(데이터프레임)
    X_test = pd.DataFrame(data["X_test"])  # test(테스트) X DataFrame(데이터프레임)
    X_train = pd.DataFrame(data["X_train"])  # train(학습) X DataFrame(데이터프레임)

    y = pd.Series(data["y"])  # 전체 target(타깃) y Series(시리즈)
    y_test = pd.Series(data["y_test"])  # test(테스트) y Series(시리즈)
    y_train = pd.Series(data["y_train"])  # train(학습) y Series(시리즈)

    best_name = data["best_name"]  # best(최적) model(모델) 이름
    best_model = ONNXModelWrapper(onnx_path)  # ONNX(온엑스) wrapper(래퍼)로 best model(모델) 로드

    model_state = {  # model_state(모델 상태) dict(딕셔너리) 구성
        "metrics": metrics,  # 평가 metrics(지표)
        "importance": importance,  # feature importance(특성 중요도)
        "predictions": predictions,  # predictions(예측) 결과
        "best_name": best_name,  # best model(모델) 이름
        "feature_columns": data["feature_columns"],  # feature(특성) 컬럼 목록
        "model_groups": data["model_groups"],  # model(모델) 그룹 정보
        "X": X,  # 전체 X
        "y": y,  # 전체 y
        "X_train": X_train,  # train(학습) X
        "y_train": y_train,  # train(학습) y
        "X_test": X_test,  # test(테스트) X
        "y_test": y_test,  # test(테스트) y
        "models": {best_name: best_model},  # best model(모델) dict(딕셔너리)
    }

    # --- 차원 축소·임베딩 precomputed(사전 계산) 결과 복원 ---
    if "precomputed_tsne_xy" in data:  # t-SNE 좌표가 있으면
        model_state["precomputed_tsne_xy"] = np.array(data["precomputed_tsne_xy"])  # t-SNE 2D 좌표 array(배열) 복원
    if "precomputed_umap_xy" in data:  # UMAP 좌표가 있으면
        model_state["precomputed_umap_xy"] = np.array(data["precomputed_umap_xy"])  # UMAP 2D 좌표 array(배열) 복원
        model_state["precomputed_umap_title"] = data.get("precomputed_umap_title", "UMAP")  # UMAP 차트 제목
    if "precomputed_cca_x_c" in data and "precomputed_cca_y_c" in data:  # CCA 좌표가 있으면
        model_state["precomputed_cca_x_c"] = np.array(data["precomputed_cca_x_c"])  # CCA x 좌표 array(배열) 복원
        model_state["precomputed_cca_y_c"] = np.array(data["precomputed_cca_y_c"])  # CCA y 좌표 array(배열) 복원

    # --- bootstrap(부트스트랩) precomputed(사전 계산) 결과 복원 ---
    if "precomputed_bootstrap" in data:  # bootstrap(부트스트랩) 결과가 있으면
        pb = data["precomputed_bootstrap"]  # bootstrap(부트스트랩) dict(딕셔너리) 추출
        model_state["precomputed_bootstrap"] = (  # tuple(튜플) 형태로 저장
            pb["ci_rmse"],  # RMSE CI(신뢰구간)
            pb["ci_r2"],  # R² CI(신뢰구간)
            np.array(pb["bootstrap_rmse"]),  # bootstrap(부트스트랩) RMSE array(배열)
            np.array(pb["bootstrap_r2"])  # bootstrap(부트스트랩) R² array(배열)
        )

    # --- nested CV(중첩 교차검증) precomputed(사전 계산) 결과 복원 ---
    if "precomputed_nested_cv" in data:  # nested CV(중첩 교차검증) 결과가 있으면
        pnc = data["precomputed_nested_cv"]  # nested CV dict(딕셔너리) 추출
        model_state["precomputed_nested_cv"] = (  # tuple(튜플) 형태로 저장
            pnc["mean_rmse"],  # 평균 RMSE
            pnc["std_rmse"],  # RMSE 표준편차
            np.array(pnc["outer_scores"])  # outer fold(겹) score(점수) array(배열)
        )

    # --- adversarial(적대적)·ablation(제거)·DCA precomputed(사전 계산) 결과 복원 ---
    if "precomputed_adversarial" in data:  # adversarial(적대적) 결과가 있으면
        model_state["precomputed_adversarial"] = pd.DataFrame(data["precomputed_adversarial"])  # DataFrame(데이터프레임) 복원

    if "precomputed_ablation" in data:  # ablation(제거) 결과가 있으면
        model_state["precomputed_ablation"] = pd.DataFrame(data["precomputed_ablation"])  # DataFrame(데이터프레임) 복원

    if "precomputed_dca" in data:  # DCA(결정 곡선 분석) 결과가 있으면
        model_state["precomputed_dca"] = pd.DataFrame(data["precomputed_dca"])  # DataFrame(데이터프레임) 복원

    # --- spatial(공간) external validation(외부 검증) precomputed(사전 계산) 결과 복원 ---
    if "precomputed_spatial" in data:  # spatial(공간) 검증 결과가 있으면
        ps = data["precomputed_spatial"]  # spatial(공간) dict(딕셔너리) 추출
        spatial_decoded = {}  # region(지역)별 metrics(지표) tuple(튜플) dict(딕셔너리)
        for region, metrics_list in ps.items():  # region(지역)별 순회
            spatial_decoded[region] = (  # metrics(지표) tuple(튜플) 구성 (4번째는 None)
                metrics_list[0],  # 첫 번째 metric(지표)
                metrics_list[1],  # 두 번째 metric(지표)
                metrics_list[2],  # 세 번째 metric(지표)
                None  # placeholder(자리표시자) (모델 객체 미포함)
            )
        model_state["precomputed_spatial"] = spatial_decoded  # spatial(공간) 검증 결과 저장

    # --- survival(생존)·ROC precomputed(사전 계산) 결과 복원 ---
    if "precomputed_survival_5" in data:  # 5% 성장률 survival(생존) 결과가 있으면
        model_state["precomputed_survival_5"] = pd.DataFrame(data["precomputed_survival_5"])  # DataFrame(데이터프레임) 복원

    if "precomputed_roc_data" in data:  # ROC(수신자 조작 특성) 데이터가 있으면
        model_state["precomputed_roc_data"] = data["precomputed_roc_data"]  # ROC dict(딕셔너리) 그대로 저장

    # --- SMOTE(합성 소수 오버샘플링) model_state(모델 상태) 복원 ---
    model_state_smote = None  # SMOTE model_state(모델 상태) 초기값 None
    if "model_state_smote_metrics" in data:  # SMOTE metrics(지표)가 있으면
        smote_metrics = pd.DataFrame(data["model_state_smote_metrics"])  # SMOTE metrics(지표) DataFrame(데이터프레임) 복원
        model_state_smote = {  # SMOTE model_state(모델 상태) dict(딕셔너리) 구성
            "metrics": smote_metrics,  # SMOTE metrics(지표)
            "best_name": data.get("model_state_smote_best_name", "RandomForest (Tuned)")  # SMOTE best model(모델) 이름
        }

    return model_state, model_state_smote  # model_state(모델 상태), SMOTE model_state(모델 상태) tuple(튜플) 반환

