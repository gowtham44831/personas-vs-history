"""Why does a segment pass or fail? python scripts/11_diagnose.py SETS.parquet RESULTS.parquet"""
import sys
import numpy as np, pandas as pd

S = pd.read_parquet(sys.argv[1])
R = pd.read_parquet(sys.argv[2])
R["p_buy"] = pd.to_numeric(R.p_buy, errors="coerce")
S["reviews"] = S.rating_number.fillna(0)
S["stars"] = S.average_rating.fillna(0)
pd.set_option("display.width", 200)

def pct(g, col):  # bought item's position within its set: 0 = lowest value, 1 = highest
    r = g[col].rank(method="average")
    return (r[g.is_bought].iloc[0] - 1) / (len(g) - 1)

def auc(g, score):
    s = score.loc[g.index]; b = s[g.is_bought].iloc[0]; n = s[~g.is_bought]
    return ((n < b).sum() + 0.5 * (n == b).sum()) / len(n)

rows = []
for sid, g in S.groupby("set_id"):
    if g.is_bought.sum() != 1:
        continue
    b = g[g.is_bought].iloc[0]
    rows.append({"set_id": sid, "segment": g.segment.iloc[0],
                 "price_pct": pct(g, "price"), "reviews_pct": pct(g, "reviews"), "stars_pct": pct(g, "stars"),
                 "bought_cheapest": float(b.price <= g.price.min()),
                 "price_vs_set_median": b.price / g.price.median(),
                 "auc_cheapest_first": auc(g, -S.price), "auc_most_reviews": auc(g, S.reviews),
                 "auc_best_stars": auc(g, S.stars)})
B = pd.DataFrame(rows)

print("1) Where the bought item sits in its set (0 = lowest in set, 1 = highest)")
print(B.groupby("segment")[["price_pct", "reviews_pct", "stars_pct", "bought_cheapest",
                             "price_vs_set_median"]].mean().round(2))

print("\n2) Simple rules as judges (AUC, 0.5 = random)")
print(B.groupby("segment")[["auc_cheapest_first", "auc_most_reviews", "auc_best_stars"]].mean().round(3))

M = R.merge(S[["set_id", "parent_asin", "price", "reviews", "stars"]], on=["set_id", "parent_asin"])
def sp(g, col):
    x = g[["p_buy", col]].dropna()
    if x.p_buy.nunique() < 2 or x[col].nunique() < 2:
        return np.nan
    return x.p_buy.corr(x[col], method="spearman")
tend = (M.groupby(["judge", "set_id"])
         .apply(lambda g: pd.Series({"follows_price": sp(g, "price"), "follows_reviews": sp(g, "reviews"),
                                     "follows_stars": sp(g, "stars")}))
         .groupby("judge").mean().round(2))
print("\n3) What each judge's scores follow (avg within-set rank correlation; negative price = prefers cheap)")
print(tend)
