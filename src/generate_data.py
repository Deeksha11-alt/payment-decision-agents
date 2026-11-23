"""Generate a SYNTHETIC payment transactions dataset.
All data is simulated (no real customer or company data).
Run from the project root:  python src/generate_data.py
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 100_000

# ---------- 1. Basic transaction fields ----------
categories = ["grocery", "restaurant", "fuel", "travel", "electronics",
              "digital_goods", "gambling", "healthcare"]
cat_probs = [0.25, 0.18, 0.12, 0.08, 0.10, 0.12, 0.05, 0.10]
category = rng.choice(categories, N, p=cat_probs)

channel = rng.choice(["pos", "contactless", "ecommerce", "recurring"], N,
                     p=[0.30, 0.25, 0.35, 0.10])
card_present = np.isin(channel, ["pos", "contactless"]).astype(int)

countries = ["GB", "DE", "FR", "US", "IN", "NG", "BR"]
card_country = rng.choice(countries, N, p=[0.30, 0.15, 0.12, 0.20, 0.13, 0.04, 0.06])
domestic = rng.random(N) < 0.88
merchant_country = np.where(domestic, card_country, rng.choice(countries, N))
cross_border = (card_country != merchant_country).astype(int)

base = {"grocery": 3.4, "restaurant": 3.3, "fuel": 3.8, "travel": 5.3,
        "electronics": 5.0, "digital_goods": 3.0, "gambling": 3.8, "healthcare": 4.0}
mu = np.array([base[c] for c in category])
amount = np.round(rng.lognormal(mean=mu, sigma=0.8), 2)

start = pd.Timestamp("2026-08-01")
timestamp = start + pd.to_timedelta(rng.integers(0, 90 * 24 * 3600, N), unit="s")
hour = timestamp.hour

card_id = rng.integers(10_000, 30_000, N)
new_device = ((channel == "ecommerce") & (rng.random(N) < 0.18)).astype(int)
txn_count_24h = rng.poisson(1.2, N)

# ---------- 2. True fraud (hidden ground truth) ----------
risky_cat = np.isin(category, ["electronics", "digital_goods", "gambling"]).astype(int)
night = (hour <= 4).astype(int)
logit = (-6.2
         + 1.5 * (channel == "ecommerce")
         + 1.2 * cross_border
         + 0.9 * risky_cat
         + 0.8 * night
         + 1.4 * new_device
         + 0.35 * np.clip(txn_count_24h - 2, 0, None)
         + 0.6 * (np.log1p(amount) - 4))
p_fraud = 1 / (1 + np.exp(-logit))
is_fraud = (rng.random(N) < p_fraud).astype(int)

# ---------- 3. Authorization decision (simplified) ----------
risk_score = np.clip(p_fraud * 6 + rng.normal(0, 0.05, N), 0, 1)
status = np.array(["approved"] * N, dtype=object)
reason = np.array([""] * N, dtype=object)

tech = rng.random(N)
m = tech < 0.025
status[m], reason[m] = "declined", "insufficient_funds"
m = (tech >= 0.025) & (tech < 0.035)
status[m], reason[m] = "declined", "expired_or_invalid_card"
m = (tech >= 0.035) & (tech < 0.045)
status[m], reason[m] = "declined", "do_not_honor"

still_ok = status == "approved"
fraud_flag = still_ok & (risk_score > 0.30) & (rng.random(N) < 0.85)
status[fraud_flag], reason[fraud_flag] = "declined", "suspected_fraud"

eea = np.isin(card_country, ["GB", "DE", "FR"])
still_ok = status == "approved"
sca = still_ok & (channel == "ecommerce") & eea & (amount > 30) & (rng.random(N) < 0.06)
status[sca], reason[sca] = "declined", "sca_required"

df = pd.DataFrame({
    "transaction_id": np.arange(1, N + 1),
    "timestamp": timestamp,
    "card_id": card_id,
    "card_country": card_country,
    "merchant_country": merchant_country,
    "cross_border": cross_border,
    "merchant_category": category,
    "channel": channel,
    "card_present": card_present,
    "amount": amount,
    "new_device": new_device,
    "txn_count_24h": txn_count_24h,
    "auth_status": status,
    "decline_reason": reason,
    "is_fraud": is_fraud,
}).sort_values("timestamp").reset_index(drop=True)

df.to_csv("data/synthetic_payments.csv", index=False)

# ---------- 4. Quick summary ----------
print("Rows:", len(df))
print("Fraud rate: {:.2%}".format(df["is_fraud"].mean()))
print("Approval rate: {:.2%}".format((df["auth_status"] == "approved").mean()))
print("\nDecline reasons:")
print(df["decline_reason"].replace("", "approved").value_counts())