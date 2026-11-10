"""Behavioural segments for one category.  CAT=Video_Games python scripts/02b_segments_v2.py"""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from cfg import PROC, ECO, eco_col

ms = lambda d: pd.Timestamp(d).value // 10**6
A0, A1, B1 = ms("2019-01-01"), ms("2022-01-01"), ms("2023-01-01")

p = pd.read_parquet(f"{PROC}/purchases.parquet", columns=["user_id", "parent_asin", "timestamp"],
                    filters=[("timestamp", ">=", A0), ("timestamp", "<", B1)])
p["period"] = np.where(p.timestamp < A1, "A", "B")
cnt = p.groupby(["user_id", "period"]).size().unstack(fill_value=0)
users = cnt[(cnt["A"] >= 3) & (cnt["B"] >= 1)].index

meta = pd.read_parquet(f"{PROC}/meta.parquet", columns=[
    "parent_asin", "title", "brand", "store", "price", "rating_number", "categories"])
meta["leaf"] = meta.categories.fillna("").str.split(" > ").str[-1].replace("", "Unknown")
meta["brand_key"] = meta.brand.fillna(meta.store).str.lower().str.strip()
meta["price_pct"] = meta.groupby("leaf").price.rank(pct=True)
meta["log_pop"] = np.log1p(meta.rating_number)
t = meta.title.fillna("").str.lower()
eco_cols = []
for name, pat in ECO.items():
    c = eco_col(name); eco_cols.append(c)
    meta[c] = t.str.contains(pat, regex=True).astype(float)

d = p[(p.period == "A") & p.user_id.isin(users)].merge(
    meta[["parent_asin", "brand_key", "price_pct", "log_pop", "leaf"] + eco_cols], on="parent_asin", how="left")
d["budget"] = (d.price_pct <= 0.25).where(d.price_pct.notna()).astype(float)
d["premium"] = (d.price_pct >= 0.75).where(d.price_pct.notna()).astype(float)

g = d.groupby("user_id")
feat = pd.DataFrame({"n_purchases": g.size(), "n_priced": g.price_pct.count(),
                     "budget_share": g.budget.mean(), "premium_share": g.premium.mean(),
                     "popularity": g.log_pop.median()})
for c in eco_cols:
    feat[c] = g[c].mean()
bc = d.dropna(subset=["brand_key"]).groupby(["user_id", "brand_key"]).size()
tot = bc.groupby(level=0).sum()
feat["brand_loyalty"] = (bc.groupby(level=0).max() / tot).where(tot >= 3)
feat = feat[feat.n_priced >= 2]
print("pilot users (>=3 purchases 2019-21, >=1 in 2022, >=2 priced):", len(feat))

cols = ["budget_share", "premium_share", "brand_loyalty"] + eco_cols + ["popularity", "n_purchases"]
X = feat[cols].copy(); X["n_purchases"] = np.log1p(X.n_purchases); X = X.fillna(X.median())
Xs = StandardScaler().fit_transform(X)
idx = np.random.default_rng(42).choice(len(Xs), size=min(10_000, len(Xs)), replace=False)
best = None
for k in range(4, 9):
    s = silhouette_score(Xs[idx], KMeans(n_clusters=k, n_init=5, random_state=42).fit(Xs[idx]).labels_)
    print(f"k={k} silhouette={s:.3f}")
    best = (k, s) if best is None or s > best[1] else best
print("chosen k:", best[0])
feat["segment"] = KMeans(n_clusters=best[0], n_init=5, random_state=42).fit_predict(Xs)

prof = feat.groupby("segment")[cols].mean().round(2)
prof["users"] = feat.groupby("segment").size(); prof["share"] = (prof.users / prof.users.sum()).round(3)
pd.set_option("display.width", 220)
print("\nsegment profiles (means):\n", prof)
d = d.merge(feat[["segment"]], left_on="user_id", right_index=True)
for s in sorted(feat.segment.unique()):
    sub = d[d.segment == s]
    print(f"\nsegment {s} | top subcategories:", sub.leaf.value_counts(normalize=True).head(4).round(2).to_dict())
    print(f"segment {s} | top brands:", sub.brand_key.value_counts(normalize=True).head(4).round(2).to_dict())
feat.reset_index().to_parquet(f"{PROC}/user_segments_v2.parquet", index=False)
prof.to_csv(f"{PROC}/segment_profiles_v2.csv")
print(f"\nsaved {PROC}/user_segments_v2.parquet — next: name the segments in config/<CAT>.json, then run 03")
