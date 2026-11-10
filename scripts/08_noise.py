"""Compare two judge runs on the same products: python scripts/08_noise.py A.parquet B.parquet [judge]"""
import sys
import pandas as pd
from sklearn.metrics import cohen_kappa_score

a, b = pd.read_parquet(sys.argv[1]), pd.read_parquet(sys.argv[2])
judge = sys.argv[3] if len(sys.argv) > 3 else "Generic shopper"
for d in (a, b):
    d["grade"] = pd.to_numeric(d["grade"], errors="coerce")
m = a[a.judge == judge].merge(b[b.judge == judge], on=["query_id", "product_id"], suffixes=("_a", "_b"))
m = m.dropna(subset=["grade_a", "grade_b"])
diff = (m.grade_a - m.grade_b).abs()
print(f"judge: {judge} | products compared: {len(m)}")
print(f"same judge differs by >=1: {(diff >= 1).mean():.0%} | by >=2: {(diff >= 2).mean():.0%}")
print(f"self-agreement kappa: {cohen_kappa_score(m.grade_a.astype(int), m.grade_b.astype(int), weights='quadratic'):.2f}")
