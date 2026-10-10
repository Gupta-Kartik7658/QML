"""Generate SMALL MOCK files with the exact PaySim / BankSim schemas — for software tests only.

These are random numbers with an injected signal; they are NOT PaySim/BankSim and must never
be used for paper results. Usage:  python tests/make_mock_data.py <out_dir> [n_rows]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def mock_paysim(n, rng):
    types = np.array(["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
    t = rng.choice(types, n, p=[0.22, 0.35, 0.01, 0.34, 0.08])
    amount = np.round(rng.lognormal(10, 1.5, n), 2)
    old_o = np.round(rng.lognormal(9, 2.5, n) * (rng.random(n) > 0.3), 2)
    fraud = ((np.isin(t, ["CASH_OUT", "TRANSFER"])) & (rng.random(n) < 0.02 + 0.2 * (amount > np.quantile(amount, .95)))).astype(int)
    new_o = np.where(fraud == 1, 0.0, np.maximum(old_o - amount, 0))
    old_d = np.round(rng.lognormal(10, 2.5, n) * (rng.random(n) > 0.4), 2)
    new_d = old_d + np.where(rng.random(n) > 0.5, amount, 0)
    dest = np.where(t == "PAYMENT", "M", "C")
    return pd.DataFrame({
        "step": rng.integers(1, 744, n), "type": t, "amount": amount,
        "nameOrig": [f"C{x}" for x in rng.integers(1e8, 1e9, n)], "oldbalanceOrg": old_o, "newbalanceOrig": new_o,
        "nameDest": [f"{d}{x}" for d, x in zip(dest, rng.integers(1e8, 1e9, n))], "oldbalanceDest": old_d,
        "newbalanceDest": new_d, "isFraud": fraud, "isFlaggedFraud": (fraud & (amount > 2e5)).astype(int)})


def mock_banksim(n, rng):
    cats = ["es_transportation", "es_food", "es_health", "es_wellnessandbeauty", "es_fashion",
            "es_barsandrestaurants", "es_hyper", "es_sportsandtoys", "es_tech", "es_home", "es_hotelservices",
            "es_otherservices", "es_contents", "es_travel", "es_leisure"]
    cat = rng.choice(cats, n)
    merch = rng.integers(0, 50, n)
    amount = np.round(rng.lognormal(3.3, 1.0, n), 2)
    risky = np.isin(cat, ["es_travel", "es_leisure", "es_sportsandtoys", "es_health"]) | (merch < 4)
    fraud = (rng.random(n) < 0.002 + 0.25 * risky * (amount > 60)).astype(int)
    q = lambda a: [f"'{x}'" for x in a]  # noqa: E731  (BankSim stores quoted strings)
    return pd.DataFrame({
        "step": rng.integers(0, 180, n), "customer": q([f"C{x}" for x in rng.integers(1e8, 1e9, n)]),
        "age": q(rng.choice(list("0123456U"), n)), "gender": q(rng.choice(list("FMEU"), n, p=[.54, .45, .005, .005])),
        "zipcodeOri": q(["28007"] * n), "merchant": q([f"M{1000 + m}" for m in merch]), "zipMerchant": q(["28007"] * n),
        "category": q(cat), "amount": amount, "fraud": fraud})


if __name__ == "__main__":
    out = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    mock_paysim(n, rng).to_csv(out / "PS_20174392719_1491204439457_log.csv", index=False)
    mock_banksim(n, rng).to_csv(out / "bs140513_032310.csv", index=False)
    print("mock files written to", out)
