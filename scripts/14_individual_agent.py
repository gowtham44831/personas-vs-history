"""Individual-context agents: segment profile + the shopper's own 2019-2021 purchase history.
Uses the cached choice sets of 09_calibration.py. Two new judges per set:
  "Own agent + history"  = the shopper's segment profile + their own history
  "Generic + history"    = a typical shopper + their own history (isolates what the segment adds)
JUDGE_MODEL=gpt-4.1-mini WINDOW=dev N_SETS=100 python scripts/14_individual_agent.py     (CAT=... for other categories)
Then: WINDOW=dev N_SETS=100 python scripts/15_individual_report.py
Options: N_RECENT (5 recent purchases shown), WORKERS (8), RESUME=1, DRY_RUN=1
"""
import os, re, json, time, threading
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from cfg import PROC, KB, RES, eco_of, DEVICE_WORD

MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1-mini")
WINDOW, N = os.getenv("WINDOW", "dev"), int(os.getenv("N_SETS", "100"))
N_RECENT, WORKERS = int(os.getenv("N_RECENT", "5")), int(os.getenv("WORKERS", "8"))
TEMP = os.getenv("JUDGE_TEMPERATURE", "0")
RESUME, DRY = os.getenv("RESUME") == "1", os.getenv("DRY_RUN") == "1"
ms = lambda d: pd.Timestamp(d).value // 10**6
A0, A1 = ms("2019-01-01"), ms("2022-01-01")
safe = re.sub(r"[^A-Za-z0-9]+", "_", MODEL)
SETS = f"{RES}/calib_sets_{WINDOW}_n{N}.parquet"
OUT = f"{RES}/calibind_{safe}_{WINDOW}_n{N}.parquet"
assert os.path.exists(SETS), f"missing {SETS}: build choice sets with 09_calibration.py first"
sets = pd.read_parquet(SETS)

# ---- each shopper's own history (Period A only: before every evaluation window) ----
H = pd.read_parquet(f"{PROC}/purchases.parquet", columns=["user_id", "parent_asin", "timestamp"],
                    filters=[("timestamp", ">=", A0), ("timestamp", "<", A1)])
H = H[H.user_id.isin(sets.user_id.unique())]
meta = pd.read_parquet(f"{PROC}/meta.parquet", columns=["parent_asin", "title", "price", "categories"])
H = H.merge(meta, on="parent_asin", how="left").dropna(subset=["title"])
H["eco"] = H.title.map(eco_of)
H["leaf"] = H.categories.fillna("").str.split(" > ").str[-1]

def history_block(uid):
    h = H[H.user_id == uid].sort_values("timestamp", ascending=False)
    if h.empty:
        return "You have no earlier purchases on record."
    eco = h.eco.value_counts(normalize=True)
    eco_txt = ", ".join(f"{round(v * 100)}% {k}" for k, v in eco.items())
    lines = [f"Your own purchase history in this category (2019-2021, {len(h)} purchases):",
             f"- The {DEVICE_WORD} you bought for: {eco_txt} (Other = not specific to one brand/platform)"]
    lines.append("- Your most recent purchases:")
    for r in h.head(N_RECENT).itertuples():
        price = f"${r.price:.2f}" if pd.notna(r.price) else "price unknown"
        lines.append(f"    {str(r.title)[:90]} | {r.leaf or 'unknown subcategory'} | {price}")
    return "\n".join(lines)

profiles = json.load(open(f"{KB}/profiles.json"))
RUBRIC = """You are shown products from one shopping category.
For each product, give p_buy: the probability (0 to 1) that you, as this shopper,
would choose to buy it rather than the others shown. Consider whether it fits the devices you own
(your own purchase history is the best evidence of that), its price compared with what you usually pay,
and your brand and quality preferences.
Judge each product on its own merits; ignore the order in which products are listed.
Return one entry per product, using its number as "i"."""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["judgments"],
          "properties": {"judgments": {"type": "array", "items": {
              "type": "object", "additionalProperties": False, "required": ["i", "p_buy"],
              "properties": {"i": {"type": "integer"}, "p_buy": {"type": "number"}}}}}}
groups = {sid: g for sid, g in sets.groupby("set_id")}

def fmt(r):
    brand = r.brand if isinstance(r.brand, str) and r.brand else (r.store or "unknown")
    rating = (f"{r.average_rating:.1f} stars, {int(r.rating_number)} ratings"
              if pd.notna(r.average_rating) and pd.notna(r.rating_number) else "rating unknown")
    feats = str(r.features or "")[:200].replace("\n", " ")
    return f"[{r.i}] {r.title} | brand: {brand} | ${r.price:.2f} | {rating} | {feats}"

def prompt(sid, jname):
    g = groups[sid]
    persona = profiles[g.segment.iloc[0]]["profile"] if jname == "Own agent + history" else "You are a typical online shopper."
    user = (history_block(g.user_id.iloc[0]) + "\n\nNow the products for this choice.\n"
            f"Shopping category: {g.leaf.iloc[0]}\n\nProducts:\n" + "\n".join(fmt(r) for r in g.itertuples()))
    return persona + "\n\n" + RUBRIC, user

JUDGES = ["Own agent + history", "Generic + history"]
if DRY:
    s, u = prompt(sets.set_id.iloc[0], JUDGES[0]); print(f"[SYSTEM]\n{s}\n\n[USER]\n{u}"); raise SystemExit

from openai import OpenAI
client = OpenAI(timeout=120)
def call(system, user):
    kw = dict(model=MODEL, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
              response_format={"type": "json_schema", "json_schema": {"name": "choices", "strict": True, "schema": SCHEMA}})
    if TEMP != "none":
        kw["temperature"] = float(TEMP)
    for attempt in range(4):
        try:
            r = client.chat.completions.create(**kw)
            return json.loads(r.choices[0].message.content)["judgments"]
        except Exception as e:
            if "temperature" in str(e) and "temperature" in kw:
                kw.pop("temperature"); continue
            if attempt == 3:
                raise
            time.sleep(2 ** attempt * 2)

results, lock, done = [], threading.Lock(), set()
if RESUME and os.path.exists(OUT):
    prev = pd.read_parquet(OUT); results = prev.to_dict("records"); done = set(zip(prev.set_id, prev.judge))
tasks = [(sid, j) for sid in sets.set_id.unique() for j in JUDGES if (sid, j) not in done]
print(f"{len(tasks)} judge calls with {WORKERS} workers")

def work(task):
    sid, jname = task; g = groups[sid]
    got = {int(j["i"]): j.get("p_buy") for j in call(*prompt(sid, jname))}
    return [{"set_id": sid, "segment": g.segment.iloc[0], "judge": jname, "i": int(r.i), "parent_asin": r.parent_asin,
             "is_bought": bool(r.is_bought), "pop_A": r.pop_A, "p_buy": got.get(int(r.i))} for r in g.itertuples()]

t0, n_done, n_fail = time.time(), 0, 0
with ThreadPoolExecutor(WORKERS) as ex:
    futs = {ex.submit(work, t): t for t in tasks}
    for f in as_completed(futs):
        try:
            rows = f.result()
            with lock:
                results.extend(rows)
        except Exception as e:
            n_fail += 1; print(f"  failed {futs[f]}: {str(e)[:120]}")
        n_done += 1
        if n_done % 100 == 0 or n_done == len(tasks):
            with lock:
                pd.DataFrame(results).to_parquet(OUT, index=False)
            print(f"{n_done}/{len(tasks)} calls | {n_fail} failed | {time.time() - t0:.0f}s")
pd.DataFrame(results).to_parquet(OUT, index=False)
print(f"\nsaved {OUT} — next: WINDOW={WINDOW} N_SETS={N} python scripts/15_individual_report.py")
