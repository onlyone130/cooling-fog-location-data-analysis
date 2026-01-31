import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import DATA_RAW, DATA_PROCESSED
from utils import get_entropy_weights, perform_topsis


def load_data():
    df_old = pd.read_csv(DATA_PROCESSED / "old_house_ratio.csv")
    df_pop = pd.read_csv(DATA_PROCESSED / "vulnerable_population.csv")
    df_heat_death = pd.read_csv(DATA_PROCESSED / "heat_death.csv")

    for df in [df_old, df_pop, df_heat_death]:
        df["구"] = df["구"].str.strip()

    return df_old, df_pop, df_heat_death


def preprocess(df_old, df_pop, df_heat_death):
    df = (
        df_old[["구", "노후화주택_비율"]]
        .merge(df_pop[["구", "0세9세생활_인구_밀도(수/km²)"]], on="구")
        .merge(df_pop[["구", "65세이상생활_인구_밀도(수/km²)"]], on="구")
        .merge(df_heat_death[["구", "23_24년_평균발생율"]], on="구")
    )
    return df


def run_vhi_topsis(df):
    cols = ["노후화주택_비율", "0세9세생활_인구_밀도(수/km²)", "65세이상생활_인구_밀도(수/km²)", "23_24년_평균발생율"]

    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[cols] = scaler.fit_transform(df[cols])

    weights = get_entropy_weights(df_scaled[cols])
    df_scaled["VHI_TOPSIS_Score"] = perform_topsis(df_scaled, cols, weights)

    result = df_scaled[["구", "VHI_TOPSIS_Score"]] \
        .sort_values(by="VHI_TOPSIS_Score", ascending=False)

    return result


if __name__ == "__main__":
    df_old, df_pop, df_single = load_data()
    df = preprocess(df_old, df_pop, df_single)

    result = run_vhi_topsis(df)
    result.to_csv(DATA_PROCESSED / "VHI_index_Topsis.csv", index=False)

    print("✅ VHI TOPSIS completed")
    print(result.head())
