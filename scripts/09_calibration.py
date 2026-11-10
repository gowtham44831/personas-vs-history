"""Calibration gate (H1): do segment agents predict what their segment actually bought?
For each segment, sample shoppers' purchases in Period B (dev = Jan-Jun 2022, test = Jul-Dec 2022).
Each choice set = the bought product + N_NEG products from the same subcategory the shopper never bought
(sampled by how often they were bought in the window, square-root damped so bestsellers don't dominate).
Every set is judged by all 4 segment agents and the generic shopper. Saves after every batch; resumable.

JUDGE_MODEL=gpt-4.1-mini N_SETS=100 WINDOW=dev python scripts/09_calibration.py
Options: N_NEG (9), WORKERS (8), RESUME=1, JUDGE_TEMPERATURE (0 | none)
"""
import os, re, json, time, threading
from cfg import PROC, KB, RES
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1-mini")
N_SETS = int(os.getenv("N_SETS", "100"))
K = int(os.getenv("N_NEG", "9"))
WORKERS = int(os.getenv("WORKERS", "8"))
WINDOW = os.getenv("WINDOW", "dev")
TEMP = os.getenv("JUDGE_TEMPERATURE", "0")
RESUME = os.getenv("RESUME") == "1"
client = OpenAI(timeout=120)

ms = lambda d: pd.Timestamp(d).value // 10**6
A0, A1, B_END = ms("2019-01-01"), ms("2022-01-01"), ms("2023-01-01")
WINDOWS = {"dev": (ms("2022-01-01"), ms("2022-07-01")), "test": (ms("2022-07-01"), ms("2023-01-01")),
           "holdout": (ms("2023-01-01"), ms("2023-10-01"))}
LO, HI = WINDOWS[WINDOW]
B_END = max(B_END, HI)

safe = re.sub(r"[^A-Za-z0-9]+", "_", MODEL)
OUT = f"{RES}/calib_{safe}_{WINDOW}_n{N_SETS}.parquet"
SETS_OUT = f"{RES}/calib_sets_{WINDOW}_n{N_SETS}.parquet"

# ---------- build choice sets (cached so every model sees the same sets) ----------
if os.path.exists(SETS_OUT):
    sets = pd.read_parquet(SETS_OUT)
    print(f"loaded {sets.set_id.nunique()} cached choice sets")
else:
    seg = pd.read_parquet(f"{PROC}/user_segments_v2.parquet", columns=["user_id", "segment_name"])
    p = pd.read_parquet(f"{PROC}/purchases.parquet", columns=["user_id", "parent_asin", "timestamp"],
                        filters=[("timestamp", ">=", A0), ("timestamp", "<", B_END)])
    meta = pd.read_parquet(f"{PROC}/meta.parquet", columns=[
        "parent_asin", "title", "brand", "store", "price", "average_rating", "rating_number",
        "categories", "features"])
    meta["leaf"] = meta.categories.fillna("").str.split(" > ").str[-1].replace("", "Unknown")
    meta = meta[(meta.leaf != "Unknown") & meta.price.notna() & meta.title.notna()]
    meta["pop_A"] = meta.parent_asin.map(p[p.timestamp < A1].parent_asin.value_counts()).fillna(0)
    meta["pop_win"] = meta.parent_asin.map(
        p[(p.timestamp >= LO) & (p.timestamp < HI)].parent_asin.value_counts()).fillna(0)
    pool = meta[meta.pop_win > 0]
    by_leaf = {k: g for k, g in pool.groupby("leaf")}

    targets = (p[(p.timestamp >= LO) & (p.timestamp < HI)]
               .merge(seg, on="user_id").merge(pool[["parent_asin", "leaf"]], on="parent_asin"))
    picked = (targets.sample(frac=1, random_state=7).drop_duplicates("user_id")
              .groupby("segment_name").head(N_SETS * 2))
    owned = p[p.user_id.isin(picked.user_id)].groupby("user_id").parent_asin.agg(set)

    rng = np.random.default_rng(7)
    rows, sid = [], 0
    for sname, g in picked.groupby("segment_name"):
        made = 0
        for _, t in g.iterrows():
            if made >= N_SETS:
                break
            cand = by_leaf[t.leaf]
            cand = cand[~cand.parent_asin.isin(owned.get(t.user_id, set()))]
            if len(cand) < K:
                continue
            w = np.sqrt(cand.pop_win.values.astype(float)); w = w / w.sum()
            neg = cand.iloc[rng.choice(len(cand), size=K, replace=False, p=w)]
            items = pd.concat([pool[pool.parent_asin == t.parent_asin].head(1), neg])
            items = items.sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True)
            items["i"] = np.arange(1, len(items) + 1)
            items["is_bought"] = items.parent_asin == t.parent_asin
            items["set_id"], items["segment"], items["user_id"] = sid, sname, t.user_id
            rows.append(items); sid += 1; made += 1
        print(f"{sname}: {made} choice sets")
    sets = pd.concat(rows, ignore_index=True)
    sets.to_parquet(SETS_OUT, index=False)

# ---------- judging ----------
RUBRIC = """You are shown products from one shopping category.
For each product, give p_buy: the probability (0 to 1) that you, as this shopper,
would choose to buy it rather than the others shown. Consider whether it fits your devices,
its price compared with what you usually pay, and your brand and quality preferences.
Judge each product on its own merits; ignore the order in which products are listed.
Return one entry per product, using its number as "i"."""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["judgments"],
          "properties": {"judgments": {"type": "array", "items": {
              "type": "object", "additionalProperties": False, "required": ["i", "p_buy"],
              "properties": {"i": {"type": "integer"}, "p_buy": {"type": "number"}}}}}}

profiles = json.load(open(f"{KB}/profiles.json"))
judges = {"Generic shopper": "You are a typical online shopper."}
judges.update({k: v["profile"] for k, v in profiles.items()})

def fmt(r):
    brand = r.brand if isinstance(r.brand, str) and r.brand else (r.store or "unknown")
    rating = (f"{r.average_rating:.1f} stars, {int(r.rating_number)} ratings"
              if pd.notna(r.average_rating) and pd.notna(r.rating_number) else "rating unknown")
    feats = str(r.features or "")[:200].replace("\n", " ")
    return f"[{r.i}] {r.title} | brand: {brand} | ${r.price:.2f} | {rating} | {feats}"

def call(system, user):
    kw = dict(model=MODEL, messages=[{"role": "system", "content": system},
                                     {"role": "user", "content": user}],
              response_format={"type": "json_schema",
                               "json_schema": {"name": "choices", "strict": True, "schema": SCHEMA}})
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

results, lock = [], threading.Lock()
done = set()
if RESUME and os.path.exists(OUT):
    prev = pd.read_parquet(OUT)
    results = prev.to_dict("records")
    REDO = [x.strip() for x in os.getenv("REDO", "").split(",") if x.strip()]
    if REDO:
        prev = prev[~prev.judge.isin(REDO)]
        results = prev.to_dict("records")
        print(f"re-judging: {REDO}")
    done = set(zip(prev.set_id, prev.judge))
    print(f"resuming: {len(done)} judgments already done")

MATCHED_ONLY = os.getenv("MATCHED_ONLY") == "1"   # only the shopper's own segment agent + generic judge
seg_of = sets.groupby("set_id").segment.first().to_dict()
tasks = [(sid, j) for sid in sets.set_id.unique() for j in judges
         if (sid, j) not in done and (not MATCHED_ONLY or j in ("Generic shopper", seg_of[sid]))]
groups = {sid: g for sid, g in sets.groupby("set_id")}
print(f"{len(tasks)} judge calls to make with {WORKERS} workers")

def work(task):
    sid, jname = task
    g = groups[sid]
    user = f"Shopping category: {g.leaf.iloc[0]}\n\nProducts:\n" + "\n".join(fmt(r) for r in g.itertuples())
    got = {int(j["i"]): j.get("p_buy") for j in call(judges[jname] + "\n\n" + RUBRIC, user)}
    return [{"set_id": sid, "segment": g.segment.iloc[0], "judge": jname, "i": int(r.i),
             "parent_asin": r.parent_asin, "is_bought": bool(r.is_bought), "pop_A": r.pop_A,
             "p_buy": got.get(int(r.i))} for r in g.itertuples()]

t0, n_done, n_fail = time.time(), 0, 0
with ThreadPoolExecutor(WORKERS) as ex:
    futs = {ex.submit(work, t): t for t in tasks}
    for f in as_completed(futs):
        try:
            rows = f.result()
            with lock:
                results.extend(rows)
        except Exception as e:
            n_fail += 1
            print(f"  failed {futs[f]}: {str(e)[:120]}")
        n_done += 1
        if n_done % 100 == 0 or n_done == len(tasks):
            with lock:
                pd.DataFrame(results).to_parquet(OUT, index=False)
            print(f"{n_done}/{len(tasks)} calls | {n_fail} failed | {time.time() - t0:.0f}s elapsed")

pd.DataFrame(results).to_parquet(OUT, index=False)
print(f"\nsaved {OUT} — run: python scripts/10_calibration_report.py {OUT}")
