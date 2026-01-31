import numpy as np

def get_entropy_weights(df):
    probs = df / df.sum(axis=0)
    probs = probs + 1e-9

    n = len(df)
    k = 1.0 / np.log(n)
    entropy = -k * (probs * np.log(probs)).sum(axis=0)

    utility = 1 - entropy
    weights = utility / utility.sum()
    return weights


def perform_topsis(df, columns, weights):
    weighted = df[columns] * weights

    pis = weighted.max()
    nis = weighted.min()

    d_plus = ((weighted - pis) ** 2).sum(axis=1) ** 0.5
    d_minus = ((weighted - nis) ** 2).sum(axis=1) ** 0.5

    score = d_minus / (d_plus + d_minus)
    return score
