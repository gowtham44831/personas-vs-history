import sys
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

path = sys.argv[1]
res = pd.read_parquet(path)
cl = pd.read_parquet(path.replace(".parquet", "_calls.parquet"))

ok = cl[cl.ok]
print(f"queries: {res.query_id.nunique()} | calls ok: {len(ok)}/{len(cl)} | "
      f"sec/call: {ok.secs.mean():.1f} | tokens in/out: {ok.tok_in.mean():.0f}/{ok.tok_out.mean():.0f}")

res["esci_grade"] = res.esci_label.map({"E": 3, "S": 2, "C": 1, "I": 0})
res["grade"] = pd.to_numeric(res["grade"], errors="coerce")
g = res.dropna(subset=["grade"]).astype({"grade": int})
print("\nagreement with ESCI human labels (E=3, S=2, C=1, I=0):")
for jname, d in g.groupby("judge"):
    k = cohen_kappa_score(d.esci_grade, d.grade, weights="quadratic")
    rho = spearmanr(d.esci_grade, d.grade).statistic
    print(f"  {jname:30s} kappa={k:.2f} spearman={rho:.2f} n={len(d)}")

seg = g[g.judge != "Generic shopper"]
spread = seg.pivot_table(index=["query_id", "product_id"], columns="judge", values="grade").dropna()
rng = spread.max(axis=1) - spread.min(axis=1)
print(f"\nproducts graded by all 4 segments: {len(spread)}")
print(f"segments differ by >=1 grade: {(rng >= 1).mean():.0%} | by >=2: {(rng >= 2).mean():.0%}")
print("\nmean grade by judge and query type:")
print(g.pivot_table(index="judge", columns="query_type", values="grade", aggfunc="mean").round(2))
