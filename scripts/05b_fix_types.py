import warnings
import pandas as pd
warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")

OUT = "data/processed"
pilot = pd.read_parquet(f"{OUT}/pilot_queries.parquet")
cands = pd.read_parquet(f"{OUT}/pilot_candidates.parquet").drop(columns=["query_type"])

q = pilot["query"].str.lower()
apple = q.str.contains(
    r"i ?phone|ipad|ipod|airpods|apple|magsafe|iwatch"
    r"|\b1[1-5] ?(pro|mini|max)\b|\bx[rs]\b|\bxs ?max\b|xrphone"
    r"|\b[6-8] ?plus\b|\b[6-8]s\b|\bse 20[12]\d\b")
samsung = q.str.contains(
    r"galaxy|samsun|\bnote ?\d|\bs\d{1,2}\b|\ba\d{2}e?\b|\bz ?flip|\bz ?fold")
other = q.str.contains(
    r"pixel|one ?plus|1\+|\b1 ?plas|\bmot\b|moto|motorola|\blg\b|stylo|huawei|xiaomi"
    r"|redmi|redimi|nokia|tcl|alcatel|\bzte\b|blu\b|kyocera|garmin|honor|xperia|sony|\blg ?k\d|lgk\d")

pilot["query_type"] = "generic"
pilot.loc[other, "query_type"] = "other_brand"
pilot.loc[samsung, "query_type"] = "samsung"
pilot.loc[apple, "query_type"] = "apple"

print(pilot.query_type.value_counts())
print("\nrandom generic sample:")
print(pilot[pilot.query_type == "generic"]["query"].sample(20, random_state=1).tolist())

pilot.to_parquet(f"{OUT}/pilot_queries.parquet", index=False)
cands.merge(pilot[["query_id", "query_type"]], on="query_id") \
     .to_parquet(f"{OUT}/pilot_candidates.parquet", index=False)
print("\nupdated pilot_queries.parquet and pilot_candidates.parquet")
