import math

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------
st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬", layout="wide")
st.title("🎬 영화 유형 나누기")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"

# 화면에 보여줄 이름과 실제 데이터 열(내부 계산용 이름)을 연결
FEATURE_LABELS = {
    "log_first_scrn": "스크린 수(로그)",
    "log_total_audi": "누적 관객수(로그)",
    "days_in_top10": "10위권 일수",
    "long_run_index": "롱런 지수",
}
LABEL_TO_KEY = {v: k for k, v in FEATURE_LABELS.items()}
ALL_LABELS = list(FEATURE_LABELS.values())
CLUSTER_ORDER = ["㉮", "㉯", "㉰"]


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """네 가지 속성을 만들고, 계산할 수 없는 영화는 제외한다."""
    df = raw_df.copy()

    needed_raw = ["first_scrn", "total_audi", "days_in_top10", "first_week_audi"]
    df = df.dropna(subset=needed_raw)

    # 첫 주 관객이 0이면 롱런 지수를 만들 수 없으므로 제외
    df = df[df["first_week_audi"] != 0]
    # 로그를 취할 수 없는(0 이하) 값도 제외
    df = df[(df["first_scrn"] > 0) & (df["total_audi"] > 0)]

    df["log_first_scrn"] = df["first_scrn"].apply(math.log10)
    df["log_total_audi"] = df["total_audi"].apply(math.log10)

    long_run = df["total_audi"] / df["first_week_audi"]
    df["long_run_index"] = long_run.apply(lambda x: min(x, 20))

    return df


# ---------------------------------------------------------
# 데이터 불러오기 및 전처리
# ---------------------------------------------------------
raw_df = load_data()
total_count = len(raw_df)

data = build_features(raw_df)
used_count = len(data)

st.write(f"전체 {total_count}편 중 묶음 분석에 사용한 영화는 {used_count}편입니다.")

# ---------------------------------------------------------
# 묶는 데 사용할 속성 선택
# ---------------------------------------------------------
st.subheader("1. 묶는 데 사용할 속성 선택")
selected_labels = st.multiselect(
    "두 개 이상 선택하세요.",
    options=ALL_LABELS,
    default=ALL_LABELS,
)

if len(selected_labels) < 2:
    st.warning("속성을 두 개 이상 선택해야 묶음을 나눌 수 있습니다.")
    st.stop()

selected_keys = [LABEL_TO_KEY[label] for label in selected_labels]

# ---------------------------------------------------------
# 표준화 + k-평균(k=3)
# ---------------------------------------------------------
X = data[selected_keys]
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data = data.copy()
data["cluster_raw"] = kmeans.fit_predict(X_scaled)

# 누적 관객 평균이 큰 묶음부터 ㉮, ㉯, ㉰ 순서로 이름 붙이기
order = (
    data.groupby("cluster_raw")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)
label_map = {raw_id: name for raw_id, name in zip(order, CLUSTER_ORDER)}
data["cluster"] = data["cluster_raw"].map(label_map)

# ---------------------------------------------------------
# 2차원 산점도
# ---------------------------------------------------------
st.subheader("2. 2차원 산점도")
col1, col2 = st.columns(2)
with col1:
    x_label = st.selectbox("가로축", selected_labels, index=0, key="axis_2d_x")
with col2:
    y_label = st.selectbox("세로축", selected_labels, index=1, key="axis_2d_y")

x_key, y_key = LABEL_TO_KEY[x_label], LABEL_TO_KEY[y_label]

fig_2d = px.scatter(
    data,
    x=x_key,
    y=y_key,
    color="cluster",
    hover_name="movieNm",
    labels={x_key: x_label, y_key: y_label, "cluster": "묶음"},
    category_orders={"cluster": CLUSTER_ORDER},
)
st.plotly_chart(fig_2d, use_container_width=True)

# ---------------------------------------------------------
# 3차원 산점도
# ---------------------------------------------------------
st.subheader("3. 3차원 산점도")
if len(selected_labels) < 3:
    st.info("3차원 산점도를 그리려면 속성을 세 개 이상 선택해야 합니다.")
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        x3_label = st.selectbox("X축", selected_labels, index=0, key="axis_3d_x")
    with c2:
        y3_label = st.selectbox("Y축", selected_labels, index=1, key="axis_3d_y")
    with c3:
        z3_label = st.selectbox("Z축", selected_labels, index=2, key="axis_3d_z")

    x3_key, y3_key, z3_key = (
        LABEL_TO_KEY[x3_label],
        LABEL_TO_KEY[y3_label],
        LABEL_TO_KEY[z3_label],
    )

    fig_3d = px.scatter_3d(
        data,
        x=x3_key,
        y=y3_key,
        z=z3_key,
        color="cluster",
        hover_name="movieNm",
        labels={
            x3_key: x3_label,
            y3_key: y3_label,
            z3_key: z3_label,
            "cluster": "묶음",
        },
        category_orders={"cluster": CLUSTER_ORDER},
    )
    fig_3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig_3d, use_container_width=True)

# ---------------------------------------------------------
# 묶음별 요약 표 (원래 단위 평균)
# ---------------------------------------------------------
st.subheader("4. 묶음별 요약")
summary_rows = []
for label in CLUSTER_ORDER:
    sub = data[data["cluster"] == label]
    summary_rows.append(
        {
            "묶음": label,
            "편수": len(sub),
            "스크린 수 평균": round(sub["first_scrn"].mean(), 1),
            "누적 관객수 평균": round(sub["total_audi"].mean(), 0),
            "10위권 일수 평균": round(sub["days_in_top10"].mean(), 1),
            "롱런 지수 평균": round(sub["long_run_index"].mean(), 2),
        }
    )
st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# 묶음별 누적 관객 상위 5편
# ---------------------------------------------------------
st.subheader("5. 묶음별 누적 관객 상위 5편")
for label in CLUSTER_ORDER:
    sub = data[data["cluster"] == label].sort_values("total_audi", ascending=False).head(5)
    st.markdown(f"**{label} 묶음**")
    for rank, title in enumerate(sub["movieNm"].tolist(), start=1):
        st.write(f"{rank}. {title}")
