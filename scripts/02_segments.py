import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

OUT = "data/processed"
ms = lambda d: pd.Timestamp(d).value // 10**6
A0, A1, B1 = ms("2019-01-01"), ms("2022-01-01"), ms("2023-01-01")

# Pilot users: >=3 purchases in A, >=1 in B
p = pd.read_parquet(f"{OUT}/purchases.parquet",
                    columns=["user_id", "parent_asin", "rating", "timestamp"])
p = p[(p.timestamp >= A0) & (p.timestamp < B1)]
p["period"] = np.where(p.timestamp < A1, "A", "B")
cnt = p.groupby(["user_id", "period"]).size().unstack(fill_value=0)
users = cnt[(cnt["A"] >= 3) & (cnt["B"] >= 1)].index
print("pilot users:", len(users))

# Period A purchases joined to product metadata
meta = pd.read_parquet(f"{OUT}/meta.parquet", columns=[
    "parent_asin", "brand", "store", "price", "average_rating", "rating_number", "categories"])
meta["leaf"] = meta.categories.fillna("").str.split(" > ").str[-1].replace("", "Unknown")
meta["brand_key"] = meta.brand.fillna(meta.store).str.lower().str.strip()
meta["price_pct"] = meta.groupby("leaf").price.rank(pct=True)
meta["log_pop"] = np.log1p(meta.rating_number)

d = p[(p.period == "A") & p.user_id.isin(users)].merge(
    meta[["parent_asin", "brand_key", "price", "price_pct", "average_rating", "log_pop", "leaf"]],
    on="parent_asin", how="left")
print("Period A purchases:", len(d), "| with price:", round(d.price.notna().mean(), 3))

# One row per user
g = d.groupby("user_id")
feat = pd.DataFrame({
    "n_purchases": g.size(),
    "price_pct": g.price_pct.mean(),
    "median_price": g.price.median(),
    "rating_given": g.rating.mean(),
    "product_quality": g.average_rating.mean(),
    "popularity": g.log_pop.median(),
    "diversity": g.leaf.nunique() / g.size(),
})
bc = d.dropna(subset=["brand_key"]).groupby(["user_id", "brand_key"]).size()
tot = bc.groupby(level=0).sum()
feat["brand_loyalty"] = (bc.groupby(level=0).max() / tot).where(tot >= 3)

cols = ["price_pct", "brand_loyalty", "rating_given", "product_quality",
        "popularity", "diversity", "n_purchases"]
X = feat[cols].copy()
X["n_purchases"] = np.log1p(X.n_purchases)
X = X.fillna(X.median())
Xs = StandardScaler().fit_transform(X)

# Choose k by silhouette on a sample
rng = np.random.default_rng(42)
idx = rng.choice(len(Xs), size=min(10_000, len(Xs)), replace=False)
best = None
for k in range(4, 8):
    km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(Xs[idx])
    s = silhouette_score(Xs[idx], km.labels_)
    print(f"k={k} silhouette={s:.3f}")
    if best is None or s > best[1]:
        best = (k, s)
k = best[0]
print("chosen k:", k)
feat["segment"] = KMeans(n_clusters=k, n_init=5, random_state=42).fit_predict(Xs)

# Segment profiles
prof = feat.groupby("segment")[cols + ["median_price"]].median().round(2)
prof["users"] = feat.groupby("segment").size()
prof["share"] = (prof.users / prof.users.sum()).round(3)
pd.set_option("display.width", 200)
print("\nsegment profiles (medians):\n", prof)

d = d.merge(feat[["segment"]], left_on="user_id", right_index=True)
for s in sorted(feat.segment.unique()):
    sub = d[d.segment == s]
    print(f"\nsegment {s} | top subcategories:",
          sub.leaf.value_counts(normalize=True).head(4).round(2).to_dict())
    print(f"segment {s} | top brands:",
          sub.brand_key.value_counts(normalize=True).head(4).round(2).to_dict())

feat.reset_index().to_parquet(f"{OUT}/user_segments.parquet", index=False)
prof.to_csv(f"{OUT}/segment_profiles.csv")
print("\nsaved user_segments.parquet and segment_profiles.csv")
