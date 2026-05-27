import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

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
    supply_total = supply_total.fillna(0)
    final = pd.merge(demand, supply_total, on=["시도", "시군구"], how="inner")
    final["전체_충전기대수"] = final["급속충전기_대수"] + final["완속충전기_대수"]
    final["전체_충전기대수"] = final["전체_충전기대수"].where(final["전체_충전기대수"] > 0, final["충전기대수"])
    final = final[(final["전체_충전기대수"] > 0) & (final["총용량_kW"] > 0)].copy()
    final["인프라_부하지수"] = final["전기차_전체대수"] / final["전체_충전기대수"]
    final["전력_부하지수"] = final["총_전력판매량"] / final["총용량_kW"]
    final["전력_부하지수"] = final["전력_부하지수"].replace([np.inf, -np.inf], np.nan).fillna(0)
    final["인프라_부하지수"] = final["인프라_부하지수"].replace([np.inf, -np.inf], np.nan).fillna(0)
    final["지역"] = final["시도"] + " " + final["시군구"]

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
