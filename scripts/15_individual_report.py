"""Compare: generic | own segment agent | generic + own history | own agent + own history, on the same choice sets."""
import os, re
import numpy as np, pandas as pd
from cfg import RES

MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1-mini"); safe = re.sub(r"[^A-Za-z0-9]+", "_", MODEL)
WINDOW, N = os.getenv("WINDOW", "dev"), int(os.getenv("N_SETS", "100"))
base = pd.read_parquet(f"{RES}/calib_{safe}_{WINDOW}_n{N}.parquet")
ind = pd.read_parquet(f"{RES}/calibind_{safe}_{WINDOW}_n{N}.parquet")
base = base[(base.judge == "Generic shopper") | (base.judge == base.segment)].copy()
base["judge"] = np.where(base.judge == "Generic shopper", "Generic", "Own segment agent")
r = pd.concat([base, ind]); r["p_buy"] = pd.to_numeric(r.p_buy, errors="coerce")

def auc(g):
    s = g.p_buy.astype(float)
    if s.isna().any() or g.is_bought.sum() != 1:
        return np.nan
    b = s[g.is_bought].iloc[0]; n = s[~g.is_bought]
    return ((n < b).sum() + 0.5 * (n == b).sum()) / len(n)
A = r.groupby(["set_id", "segment", "judge"]).apply(auc).rename("auc").reset_index().dropna()
W = A.pivot_table(index=["set_id", "segment"], columns="judge", values="auc").reset_index()
order = ["Generic", "Own segment agent", "Generic + history", "Own agent + history"]
pd.set_option("display.width", 220)
t = W.groupby("segment")[order].mean(); t.loc["ALL"] = W[order].mean()
print(f"Mean AUC of the bought item, window={WINDOW} (0.5 = random)\n"); print(t.round(3))

rng = np.random.default_rng(0)
def boot(d):
    d = d.dropna().values; m = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    return f"{d.mean():+.3f} [{np.percentile(m, 2.5):+.3f},{np.percentile(m, 97.5):+.3f}]"
comps = [("Own agent + history", "Own segment agent", "value of own history for the segment agent"),
         ("Own agent + history", "Generic + history", "value of the segment, given own history"),
         ("Generic + history", "Generic", "value of own history for the generic judge"),
         ("Generic + history", "Own segment agent", "H2: individual history vs segment persona"),
         ("Own agent + history", "Generic", "full system vs generic judge")]
print("\nDifferences (mean AUC, 95% CI):")
for a, b, label in comps:
    print(f"\n  {a} minus {b}  ({label})")
    for s, g in W.groupby("segment"):
        print(f"    {s:32s} {boot(g[a] - g[b])}")
    print(f"    {'ALL':32s} {boot(W[a] - W[b])}")
