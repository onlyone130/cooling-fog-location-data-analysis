import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import DATA_PROCESSED


def load_indices():
    uvi = pd.read_csv(DATA_PROCESSED / "UVI_index_Topsis.csv")
    vhi = pd.read_csv(DATA_PROCESSED / "VHI_index_Topsis.csv")
    return uvi, vhi


def combine_indices(uvi, vhi):
    df = uvi.merge(vhi, on="구")

    scaler = MinMaxScaler()
    df[["UVI_TOPSIS_Score", "VHI_TOPSIS_Score"]] = scaler.fit_transform(
        df[["UVI_TOPSIS_Score", "VHI_TOPSIS_Score"]]
    )

    df["Final_Heat_Vulnerability"] = (
        df["UVI_TOPSIS_Score"] + df["VHI_TOPSIS_Score"]
    ) / 2

    return df.sort_values(
        by="Final_Heat_Vulnerability", ascending=False
    )


if __name__ == "__main__":
    uvi, vhi = load_indices()
    final = combine_indices(uvi, vhi)

    final.to_csv(
        DATA_PROCESSED / "Final_Heat_Vulnerability_Index.csv",
        index=False
    )

    print("✅ Final Heat Vulnerability Index created")
    print(final.head())
