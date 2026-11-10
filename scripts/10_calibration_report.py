"""AUC of the bought item vs the others in each choice set, per shopper segment and judge."""
import sys
import numpy as np, pandas as pd

r = pd.read_parquet(sys.argv[1])
r["p_buy"] = pd.to_numeric(r.p_buy, errors="coerce")

def set_auc(g, col):
    s = g[col].astype(float)
    if s.isna().any() or g.is_bought.sum() != 1:
        return np.nan
    b = s[g.is_bought].iloc[0]; n = s[~g.is_bought]
    return ((n < b).sum() + 0.5 * (n == b).sum()) / len(n)

auc = r.groupby(["set_id", "segment", "judge"]).apply(lambda g: set_auc(g, "p_buy")).rename("auc").reset_index()
pop = (r[r.judge == "Generic shopper"].groupby(["set_id", "segment"])
       .apply(lambda g: set_auc(g, "pop_A")).rename("auc").reset_index())
pop["judge"] = "Popularity (Period A)"
auc = pd.concat([auc, pop]).dropna()

pd.set_option("display.width", 200)
print("Mean AUC of the bought item (0.5 = random). Rows = whose purchase; columns = who judged.\n")
order = ["Generic shopper", "Popularity (Period A)"] + sorted(
    j for j in auc.judge.unique() if j not in ("Generic shopper", "Popularity (Period A)"))
print(auc.pivot_table(index="segment", columns="judge", values="auc", aggfunc="mean")[order].round(3))
print("\nsets per segment:", auc[auc.judge == "Generic shopper"].groupby("segment").size().to_dict())

w = auc.pivot_table(index=["set_id", "segment"], columns="judge", values="auc").reset_index()
segs = [j for j in order[2:]]
w["matched"] = w.apply(lambda x: x[x.segment], axis=1)
w["mismatched"] = w.apply(lambda x: np.mean([x[s] for s in segs if s != x.segment]), axis=1)

rng = np.random.default_rng(0)
def boot(d):
    d = d.dropna().values
    if len(d) == 0:
        return (float("nan"),) * 3
    m = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)

print("\nGate test per segment: matched agent minus baseline (mean AUC difference, 95% CI)")
for s, g in w.groupby("segment"):
    a = boot(g.matched - g["Generic shopper"]); b = boot(g.matched - g["Popularity (Period A)"])
    c = boot(g.matched - g.mismatched)
    passed = a[1] > 0 and b[1] > 0
    print(f"  {s:30s} vs generic {a[0]:+.3f} [{a[1]:+.3f},{a[2]:+.3f}] | vs popularity {b[0]:+.3f} "
          f"[{b[1]:+.3f},{b[2]:+.3f}] | vs other segments {c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}] | "
          f"{'PASS' if passed else 'not passed'}")
a = boot(w.matched - w["Generic shopper"]); c = boot(w.matched - w.mismatched)
print(f"\nOverall: matched vs generic {a[0]:+.3f} [{a[1]:+.3f},{a[2]:+.3f}] | "
      f"matched vs other segments {c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]")

# ---------- Hybrid: each judge's score combined 50/50 with popularity (within-set ranks) ----------
h = r.copy()
h["rank_agent"] = h.groupby(["set_id", "judge"]).p_buy.rank(pct=True)
h["rank_pop"] = h.groupby(["set_id", "judge"]).pop_A.rank(pct=True)
h["hybrid"] = 0.5 * h.rank_agent + 0.5 * h.rank_pop
hauc = (h.groupby(["set_id", "segment", "judge"]).apply(lambda g: set_auc(g, "hybrid"))
         .rename("auc").reset_index().dropna())

print("\n================ HYBRID: judge + popularity (50/50) ================")
print(hauc.pivot_table(index="segment", columns="judge", values="auc", aggfunc="mean")[order[:1] + order[2:]].round(3))

hw = hauc.pivot_table(index=["set_id", "segment"], columns="judge", values="auc").reset_index()
hw = hw.merge(w[["set_id", "Popularity (Period A)"]], on="set_id")
hw["matched"] = hw.apply(lambda x: x[x.segment], axis=1)
hw["mismatched"] = hw.apply(lambda x: np.mean([x[s] for s in segs if s != x.segment]), axis=1)

print("\nHybrid gate per segment: matched+pop minus baseline (mean AUC difference, 95% CI)")
for s, g in hw.groupby("segment"):
    a = boot(g.matched - g["Generic shopper"]); b = boot(g.matched - g["Popularity (Period A)"])
    c = boot(g.matched - g.mismatched)
    passed = a[1] > 0 and b[1] > 0
    print(f"  {s:30s} vs generic+pop {a[0]:+.3f} [{a[1]:+.3f},{a[2]:+.3f}] | vs popularity {b[0]:+.3f} "
          f"[{b[1]:+.3f},{b[2]:+.3f}] | vs other segs+pop {c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}] | "
          f"{'PASS' if passed else 'not passed'}")
a = boot(hw.matched - hw["Generic shopper"]); b = boot(hw.matched - hw["Popularity (Period A)"])
print(f"\nOverall hybrid: matched+pop vs generic+pop {a[0]:+.3f} [{a[1]:+.3f},{a[2]:+.3f}] | "
      f"vs popularity alone {b[0]:+.3f} [{b[1]:+.3f},{b[2]:+.3f}]")
