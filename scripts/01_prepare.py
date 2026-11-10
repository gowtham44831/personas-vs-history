"""Raw Amazon Reviews 2023 + ESCI -> processed tables for one category.  CAT=Video_Games python scripts/01_prepare.py"""
import gzip, json
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from cfg import CAT, RAW, PROC

# 1. Verified purchases (streamed, written in chunks) + variant -> parent map
cols = ["user_id", "asin", "parent_asin", "rating", "timestamp", "helpful_vote"]
writer, buf, asin_map = None, [], {}
def flush():
    global writer, buf
    t = pa.Table.from_pylist(buf)
    writer = writer or pq.ParquetWriter(f"{PROC}/purchases.parquet", t.schema)
    writer.write_table(t); buf = []
with gzip.open(f"{RAW}/amazon2023/reviews/{CAT}.jsonl.gz", "rt") as f:
    for line in f:
        r = json.loads(line)
        asin_map[r["asin"]] = r["parent_asin"]
        if r.get("verified_purchase"):
            buf.append({c: r.get(c) for c in cols})
            if len(buf) >= 1_000_000:
                flush()
if buf:
    flush()
writer.close()
pd.DataFrame(list(asin_map.items()), columns=["asin", "parent_asin"]).to_parquet(f"{PROC}/asin_map.parquet", index=False)
print("purchases written | asin map:", len(asin_map))

# 2. Product metadata (slim)
rows = []
with gzip.open(f"{RAW}/amazon2023/meta/meta_{CAT}.jsonl.gz", "rt") as f:
    for line in f:
        m = json.loads(line); d = m.get("details") or {}
        rows.append({"parent_asin": m["parent_asin"], "title": m.get("title"), "store": m.get("store"),
                     "brand": d.get("Brand"), "price": m.get("price"),
                     "average_rating": m.get("average_rating"), "rating_number": m.get("rating_number"),
                     "categories": " > ".join(m.get("categories") or []),
                     "features": " | ".join(m.get("features") or [])})
meta = pd.DataFrame(rows)
meta["price"] = pd.to_numeric(meta["price"], errors="coerce")
meta.to_parquet(f"{PROC}/meta.parquet", index=False)
print("meta:", meta.shape)

# 3. ESCI US examples + products joined to this category's metadata
esci_dir = f"{RAW}/esci-data/shopping_queries_dataset"
ex = pd.read_parquet(f"{esci_dir}/shopping_queries_dataset_examples.parquet")
pr = pd.read_parquet(f"{esci_dir}/shopping_queries_dataset_products.parquet")
esci = ex[ex.product_locale == "us"].merge(pr[pr.product_locale == "us"], on=["product_id", "product_locale"], how="left")
amap = pd.read_parquet(f"{PROC}/asin_map.parquet")
esci = esci.merge(amap, left_on="product_id", right_on="asin", how="left")
esci["parent_asin"] = esci["parent_asin"].fillna(esci["product_id"])
esci = esci.drop(columns=["asin"]).merge(meta, on="parent_asin", how="left")
esci.to_parquet(f"{PROC}/esci_us.parquet", index=False)
print("esci_us:", esci.shape, "| rows with this category's metadata:", esci.title.notna().sum())
