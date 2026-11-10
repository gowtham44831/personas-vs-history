"""Personal rules from each shopper's own 2019-2021 history vs the LLM segment agents, on the same choice sets.
No API calls.   WINDOW=test N_SETS=200 python scripts/13_rules.py      (CAT=... for other categories)
Optional: LLM_RESULTS=<path to calib_*.parquet> (defaults to the gpt-4.1-mini file for the window)."""
import os
import numpy as np, pandas as pd
from cfg import PROC, RES, eco_of

WINDOW = os.getenv("WINDOW", "test"); N = int(os.getenv("N_SETS", "200"))
SETS = f"{RES}/calib_sets_{WINDOW}_n{N}.parquet"
LLM = os.getenv("LLM_RESULTS", f"{RES}/calib_gpt_4_1_mini_{WINDOW}_n{N}.parquet")
ms = lambda d: pd.Timestamp(d).value // 10**6
A0, A1 = ms("2019-01-01"), ms("2022-01-01")

S = pd.read_parquet(SETS)
meta = pd.read_parquet(f"{PROC}/meta.parquet", columns=["parent_asin", "title", "brand", "store", "price", "categories"])
meta["leaf"] = meta.categories.fillna("").str.split(" > ").str[-1].replace("", "Unknown")
meta["price_pct"] = meta.groupby("leaf").price.rank(pct=True)
meta["brand_key"] = meta.brand.fillna(meta.store).str.lower().str.strip()

H = pd.read_parquet(f"{PROC}/purchases.parquet", columns=["user_id", "parent_asin", "timestamp"],
                    filters=[("timestamp", ">=", A0), ("timestamp", "<", A1)])
H = H[H.user_id.isin(S.user_id.unique())]
need = set(H.parent_asin) | set(S.parent_asin)
m = meta[meta.parent_asin.isin(need)].copy()
m["eco"] = m.title.map(eco_of)
cols = ["parent_asin", "price_pct", "brand_key", "eco"]
H = H.merge(m[cols], on="parent_asin", how="left")
print(f"{S.set_id.nunique()} choice sets | {S.user_id.nunique()} shoppers | {len(H):,} history purchases (2019-21)")

user_pct = H.groupby("user_id").price_pct.mean()
user_eco = H.groupby("user_id").eco.value_counts(normalize=True).to_dict()
user_brands = H.dropna(subset=["brand_key"]).groupby("user_id").brand_key.agg(set).to_dict()

X = S.merge(m[cols].rename(columns={"price_pct": "item_pct"}), on="parent_asin", how="left")
X["Cheapest"] = -X.price
X["Most reviews"] = X.rating_number.fillna(0)
X["Value for money"] = X.average_rating.fillna(0) * np.log1p(X.rating_number.fillna(0)) / X.price
X["Popularity"] = X.pop_A
X["Price fit (own history)"] = -(X.item_pct - X.user_id.map(user_pct)).abs()
X["Ecosystem match (own history)"] = [user_eco.get((u, e), 0.0) for u, e in zip(X.user_id, X.eco)]
X["Brand match (own history)"] = [float(b in user_brands.get(u, set())) for u, b in zip(X.user_id, X.brand_key)]
rk = lambda c: X.groupby("set_id")[c].rank(pct=True)
X["Combined rule (eco + price fit + popularity)"] = (
    rk("Ecosystem match (own history)") + rk("Price fit (own history)") + rk("Popularity")) / 3

rules = ["Cheapest", "Most reviews", "Value for money", "Popularity", "Price fit (own history)",
         "Ecosystem match (own history)", "Brand match (own history)", "Combined rule (eco + price fit + popularity)"]

if os.path.exists(LLM):
    R = pd.read_parquet(LLM); R["p_buy"] = pd.to_numeric(R.p_buy, errors="coerce")
    gen = R[R.judge == "Generic shopper"][["set_id", "parent_asin", "p_buy"]].rename(columns={"p_buy": "LLM generic judge"})
    own = R[R.judge == R.segment][["set_id", "parent_asin", "p_buy"]].rename(columns={"p_buy": "LLM own segment agent"})
    X = X.merge(gen, on=["set_id", "parent_asin"], how="left").merge(own, on=["set_id", "parent_asin"], how="left")
    X["Own agent + combined rule"] = (rk("LLM own segment agent") + X["Combined rule (eco + price fit + popularity)"]) / 2
    rules += ["LLM generic judge", "LLM own segment agent", "Own agent + combined rule"]
else:
    print(f"(no LLM results at {LLM}; showing rules only)")

def auc(g, c):
    s = g[c].astype(float).fillna(-1e12)
    b = s[g.is_bought].iloc[0]; n = s[~g.is_bought]
    return ((n < b).sum() + 0.5 * (n == b).sum()) / len(n)

rows = [{"set_id": sid, "segment": g.segment.iloc[0], **{r: auc(g, r) for r in rules}}
        for sid, g in X.groupby("set_id") if g.is_bought.sum() == 1]
A = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
tab = A.groupby("segment")[rules].mean().T
tab["ALL"] = A[rules].mean()
print(f"\nMean AUC of the bought item, window={WINDOW} (0.5 = random). Rows = scoring method; columns = whose purchase.\n")
print(tab.round(3))

rng = np.random.default_rng(0)
def boot(d):
    d = d.dropna().values
    m_ = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    return d.mean(), np.percentile(m_, 2.5), np.percentile(m_, 97.5)
if "LLM own segment agent" in A:
    print("\nOwn segment agent minus combined personal rule (95% CI); positive = LLM agent better")
    for s, g in A.groupby("segment"):
        a = boot(g["LLM own segment agent"] - g["Combined rule (eco + price fit + popularity)"])
        c = boot(g["Own agent + combined rule"] - g["Combined rule (eco + price fit + popularity)"])
        print(f"  {s:30s} agent - rule {a[0]:+.3f} [{a[1]:+.3f},{a[2]:+.3f}] | agent+rule - rule {c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]")
    a = boot(A["LLM own segment agent"] - A["Combined rule (eco + price fit + popularity)"])
    c = boot(A["Own agent + combined rule"] - A["Combined rule (eco + price fit + popularity)"])
    print(f"  {'ALL':30s} agent - rule {a[0]:+.3f} [{a[1]:+.3f},{a[2]:+.3f}] | agent+rule - rule {c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]")
