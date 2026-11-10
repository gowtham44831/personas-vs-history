"""ESCI queries well covered by this category's metadata, tagged by query type.  CAT=Video_Games python scripts/05_pilot_queries.py"""
import warnings, pandas as pd
from cfg import PROC, QTYPES
warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")

cols = ["query_id", "query", "product_id", "parent_asin", "esci_label", "product_title", "product_brand",
        "product_bullet_point", "product_description", "title", "brand", "price", "average_rating",
        "rating_number", "categories"]
esci = pd.read_parquet(f"{PROC}/esci_us.parquet", columns=cols)
cov = esci.groupby("query_id").agg(query=("query", "first"), n=("product_id", "size"),
                                   matched=("title", lambda s: s.notna().sum()))
cov["share"] = cov.matched / cov.n
pilot = cov[(cov.share >= 0.5) & (cov.n >= 8)].copy()
q = pilot["query"].str.lower()
pilot["query_type"] = "generic"
for name, pat in QTYPES:
    pilot.loc[q.str.contains(pat, regex=True), "query_type"] = name
print("pilot queries:", len(pilot)); print(pilot.query_type.value_counts())
for t in pilot.query_type.unique():
    print(f"\n{t} examples:", pilot[pilot.query_type == t]["query"].head(8).tolist())
pilot = pilot.reset_index()
pilot.to_parquet(f"{PROC}/pilot_queries.parquet", index=False)
esci[esci.query_id.isin(pilot.query_id)].merge(pilot[["query_id", "query_type"]], on="query_id") \
    .to_parquet(f"{PROC}/pilot_candidates.parquet", index=False)
print(f"\nsaved {PROC}/pilot_queries.parquet and pilot_candidates.parquet")
