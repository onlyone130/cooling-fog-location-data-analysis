import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import  DATA_RAW, DATA_PROCESSED
from utils import get_entropy_weights, perform_topsis


def load_data():
    df_t = pd.read_csv(DATA_PROCESSED / "gu_temperature.csv")
    df_r = pd.read_csv(DATA_PROCESSED / "road_ratio.csv")
    df_b = pd.read_csv(DATA_PROCESSED / "building_density.csv")
    df_p = pd.read_csv(DATA_PROCESSED / "daytime_population_density.csv")

    for df in [df_t, df_r, df_b, df_p]:
        df["구"] = df["구"].str.strip()

    return df_t, df_r, df_b, df_p


def preprocess(df_t, df_r, df_b, df_p):
    df = (
        df_t[["구", "기온(°C)", "체감온도"]]
        .merge(df_r[["구", "도로비율(%)"]], on="구")
        .merge(df_b[["구", "건물밀도(%)"]], on="구")
        .merge(df_p[["구", "주간인구밀도(수/km²)"]], on="구")
    )
    return df


def run_uvi_topsis(df):
    cols = ["기온(°C)", "체감온도", "도로비율(%)", "건물밀도(%)", "주간인구밀도(수/km²)"]

    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[cols] = scaler.fit_transform(df[cols])

    weights = get_entropy_weights(df_scaled[cols])
    df_scaled["UVI_TOPSIS_Score"] = perform_topsis(df_scaled, cols, weights)

    result = df_scaled[["구", "UVI_TOPSIS_Score"]] \
        .sort_values(by="UVI_TOPSIS_Score", ascending=False)

    return result


if __name__ == "__main__":
    df_t, df_r, df_b, df_p = load_data()
    df = preprocess(df_t, df_r, df_b, df_p)

    result = run_uvi_topsis(df)
    result.to_csv(DATA_PROCESSED / "UVI_index_Topsis.csv", index=False)

    print("✅ UVI TOPSIS completed")
    print(result.head())


# print(df)