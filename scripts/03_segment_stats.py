"""Segment statistics (knowledge base, layer 1).  CAT=Video_Games python scripts/03_segment_stats.py"""
import json, numpy as np, pandas as pd
from cfg import PROC, KB, ECO, SEG_NAMES, eco_col

ms = lambda d: pd.Timestamp(d).value // 10**6
A0, A1, B1 = ms("2019-01-01"), ms("2022-01-01"), ms("2023-01-01")
seg = pd.read_parquet(f"{PROC}/user_segments_v2.parquet")
names = {s: SEG_NAMES.get(int(s), f"Segment {s}") for s in sorted(seg.segment.unique())}
if not SEG_NAMES:
    print("WARNING: no segment_names in config yet; using 'Segment N'. Name them and rerun 03 and 04.")
seg["segment_name"] = seg.segment.map(names)
seg.to_parquet(f"{PROC}/user_segments_v2.parquet", index=False)

p = pd.read_parquet(f"{PROC}/purchases.parquet", columns=["user_id", "parent_asin", "rating", "timestamp"],
                    filters=[("timestamp", ">=", A0), ("timestamp", "<", B1)])
p = p[p.user_id.isin(seg.user_id)]
p["period"] = np.where(p.timestamp < A1, "A", "B")
p = p.merge(seg[["user_id", "segment"]], on="user_id")
meta = pd.read_parquet(f"{PROC}/meta.parquet", columns=["parent_asin", "brand", "store", "price", "rating_number", "categories"])
meta["leaf"] = meta.categories.fillna("").str.split(" > ").str[-1].replace("", "Unknown")
meta["brand_key"] = meta.brand.fillna(meta.store).str.lower().str.strip()
pa_ = p[p.period == "A"].merge(meta, on="parent_asin", how="left")

stats = {}
for s, name in names.items():
    u, a = seg[seg.segment == s], pa_[pa_.segment == s]
    leaves = a.leaf.value_counts().head(6).index
    bands = {leaf: {q: round(float(a.loc[a.leaf == leaf, "price"].quantile(v)), 2)
                    for q, v in [("p25", .25), ("p50", .5), ("p75", .75)]}
             for leaf in leaves if a.loc[a.leaf == leaf, "price"].notna().sum() >= 50}
    stats[name] = {
        "segment_id": int(s), "users": int(len(u)), "share_of_pilot_users": round(len(u) / len(seg), 3),
        "budget_share": round(float(u.budget_share.mean()), 2), "premium_share": round(float(u.premium_share.mean()), 2),
        "eco_shares": {e: round(float(u[eco_col(e)].mean()), 2) for e in ECO},
        "brand_loyalty": round(float(u.brand_loyalty.mean()), 2),
        "avg_rating_given": round(float(a.rating.mean()), 2),
        "median_product_reviews": int(a.rating_number.median()),
        "top_subcategories": a.leaf.value_counts(normalize=True).head(6).round(3).to_dict(),
        "top_brands": a.brand_key.value_counts(normalize=True).head(8).round(3).to_dict(),
        "price_bands_paid": bands,
        "period_A_purchases": int(len(a)),
        "period_B_purchases": int(((p.period == "B") & (p.segment == s)).sum())}
json.dump(stats, open(f"{KB}/segment_stats.json", "w"), indent=2)
print({k: (v["users"], v["eco_shares"], v["period_B_purchases"]) for k, v in stats.items()})
print(f"saved {KB}/segment_stats.json")
