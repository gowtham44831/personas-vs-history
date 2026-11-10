"""Calibration gate with few-shot examples from the knowledge base's example layer.
Uses the SAME cached choice sets as 09_calibration.py. Before judging, each agent sees N_EX past
choice situations from Period A (2019-2021, before both evaluation windows) in the same subcategory:
a product a shopper bought plus N_ALT products from that subcategory they passed over.
Segment agents get examples from their own segment's shoppers; the generic judge gets examples
from all shoppers (a fair baseline: same amount of evidence, no segment knowledge).
The target shopper is never used as an example.

DRY_RUN=1 WINDOW=dev N_SETS=100 python scripts/12_fewshot_calibration.py   (prints one prompt, no calls)
JUDGE_MODEL=gpt-4.1-mini WINDOW=dev N_SETS=100 python scripts/12_fewshot_calibration.py
Options: N_EX (3), N_ALT (4), WORKERS (8), RESUME=1
"""
import os, re, json, time, threading, zlib
from cfg import PROC, KB, RES
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1-mini")
N_SETS = int(os.getenv("N_SETS", "100"))
WINDOW = os.getenv("WINDOW", "dev")
N_EX, N_ALT = int(os.getenv("N_EX", "3")), int(os.getenv("N_ALT", "4"))
WORKERS = int(os.getenv("WORKERS", "8"))
TEMP = os.getenv("JUDGE_TEMPERATURE", "0")
RESUME, DRY = os.getenv("RESUME") == "1", os.getenv("DRY_RUN") == "1"
EX_MODE = os.getenv("EX_MODE", "random")   # random | similar
CHECK = "\u2714"

ms = lambda d: pd.Timestamp(d).value // 10**6
A0, A1 = ms("2019-01-01"), ms("2022-01-01")
safe = re.sub(r"[^A-Za-z0-9]+", "_", MODEL)
SETS = f"{RES}/calib_sets_{WINDOW}_n{N_SETS}.parquet"
OUT = (f"{RES}/calibfs_{safe}_{WINDOW}_n{N_SETS}_ex{N_EX}" + ("_sim" if EX_MODE == "similar" else "")
       + ("_matched" if os.getenv("MATCHED_ONLY") == "1" else "") + ".parquet")
assert os.path.exists(SETS), f"run 09_calibration.py first to build {SETS}"
sets = pd.read_parquet(SETS)
leaves = set(sets.leaf)

# ---------- example bank from Period A ----------
seg = pd.read_parquet(f"{PROC}/user_segments_v2.parquet", columns=["user_id", "segment_name"])
p = pd.read_parquet(f"{PROC}/purchases.parquet", columns=["user_id", "parent_asin", "timestamp"],
                    filters=[("timestamp", ">=", A0), ("timestamp", "<", A1)])
meta = pd.read_parquet(f"{PROC}/meta.parquet", columns=[
    "parent_asin", "title", "brand", "store", "price", "average_rating", "rating_number", "categories"])
meta["leaf"] = meta.categories.fillna("").str.split(" > ").str[-1]
meta = meta[meta.leaf.isin(leaves) & meta.price.notna() & meta.title.notna()].drop_duplicates("parent_asin")
p = p.merge(meta[["parent_asin", "leaf"]], on="parent_asin")
meta["pop_A"] = meta.parent_asin.map(p.parent_asin.value_counts()).fillna(0)
pool_by_leaf = {k: g[g.pop_A > 0] for k, g in meta.groupby("leaf")}
from cfg import eco_of
meta["eco"] = meta.title.map(eco_of)
pool_by_leaf = {k: g[g.pop_A > 0] for k, g in meta.groupby("leaf")}
meta_idx = meta.set_index("parent_asin")
p = p.merge(seg, on="user_id", how="left").merge(meta[["parent_asin", "eco", "price"]], on="parent_asin")

bank = {}
for leaf, g in p.groupby("leaf"):
    bank[("Generic shopper", leaf)] = g.sample(min(len(g), 3000), random_state=11)
    for s, gs in g.dropna(subset=["segment_name"]).groupby("segment_name"):
        bank[(s, leaf)] = gs.sample(min(len(gs), 3000), random_state=11)
print(f"example bank: {len(bank)} (judge, subcategory) pools from {len(p):,} Period A purchases")

def short(r):
    brand = r.brand if isinstance(r.brand, str) and r.brand else (r.store if isinstance(r.store, str) else "unknown")
    rating = (f"{r.average_rating:.1f} stars, {int(r.rating_number)} ratings"
              if pd.notna(r.average_rating) and pd.notna(r.rating_number) else "rating unknown")
    return f"{str(r.title)[:80]} | {brand} | ${r.price:.2f} | {rating}"

def examples(judge, leaf, exclude_user, seed, cur=None):
    b = bank.get((judge, leaf))
    if b is None:
        return ""
    b = b[b.user_id != exclude_user]
    if len(b) == 0:
        return ""
    r = np.random.default_rng(seed)
    pool = pool_by_leaf[leaf]
    who = "typical shoppers" if judge == "Generic shopper" else "shoppers in your segment"
    out = [f"Past choices by {who} in {leaf} ({CHECK} = the product they bought; "
           f"the others are products from the same category they did not buy):"]
    lo_p = hi_p = None
    if EX_MODE == "similar" and cur is not None:
        med = float(cur.price.median()); lo_p, hi_p = 0.5 * med, 2 * med
        eco_share = cur.title.map(eco_of).value_counts(normalize=True)
        in_band = ((b.price >= lo_p) & (b.price <= hi_p)).values
        wt = (0.1 + b.eco.map(eco_share).fillna(0).values) * np.where(in_band, 1.0, 0.2)
        wt = wt / wt.sum()
        n = min(N_EX, int((wt > 0).sum()))
        picks = b.iloc[r.choice(len(b), size=n, replace=False, p=wt)]
    else:
        picks = b.iloc[r.choice(len(b), size=min(N_EX, len(b)), replace=False)]
    for k, pk in enumerate(picks.itertuples(), 1):
        alts = pool[pool.parent_asin != pk.parent_asin]
        if lo_p is not None:
            banded = alts[(alts.price >= lo_p) & (alts.price <= hi_p)]
            if len(banded) >= N_ALT:
                alts = banded
        if len(alts) < N_ALT:
            continue
        w = np.sqrt(alts.pop_A.values.astype(float)); w = w / w.sum()
        ids = [pk.parent_asin] + list(alts.parent_asin.values[r.choice(len(alts), size=N_ALT, replace=False, p=w)])
        ex = meta_idx.loc[ids].reset_index()
        ex["bought"] = [True] + [False] * N_ALT
        ex = ex.sample(frac=1, random_state=int(r.integers(1e9)))
        out.append(f"Example {k}:")
        out += [f"  {CHECK if e.bought else ' '} {short(e)}" for e in ex.itertuples()]
    return "\n".join(out)

RUBRIC = """You are shown products from one shopping category.
For each product, give p_buy: the probability (0 to 1) that you, as this shopper,
would choose to buy it rather than the others shown. Consider whether it fits your devices,
its price compared with what you usually pay, and your brand and quality preferences.
Use the past choices you are shown as evidence of what shoppers like you actually value.
Judge each product on its own merits; ignore the order in which products are listed.
Return one entry per product, using its number as "i"."""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["judgments"],
          "properties": {"judgments": {"type": "array", "items": {
              "type": "object", "additionalProperties": False, "required": ["i", "p_buy"],
              "properties": {"i": {"type": "integer"}, "p_buy": {"type": "number"}}}}}}

profiles = json.load(open(f"{KB}/profiles.json"))
judges = {"Generic shopper": "You are a typical online shopper."}
judges.update({k: v["profile"] for k, v in profiles.items()})
groups = {sid: g for sid, g in sets.groupby("set_id")}

def fmt(r):
    brand = r.brand if isinstance(r.brand, str) and r.brand else (r.store or "unknown")
    rating = (f"{r.average_rating:.1f} stars, {int(r.rating_number)} ratings"
              if pd.notna(r.average_rating) and pd.notna(r.rating_number) else "rating unknown")
    feats = str(r.features or "")[:200].replace("\n", " ")
    return f"[{r.i}] {r.title} | brand: {brand} | ${r.price:.2f} | {rating} | {feats}"

def prompt(sid, jname):
    g = groups[sid]
    ex = examples(jname, g.leaf.iloc[0], g.user_id.iloc[0], zlib.crc32(f"{sid}|{jname}".encode()), cur=g)
    user = (ex + "\n\nNow the products for this choice.\n" if ex else "") + \
           f"Shopping category: {g.leaf.iloc[0]}\n\nProducts:\n" + "\n".join(fmt(r) for r in g.itertuples())
    return judges[jname] + "\n\n" + RUBRIC, user

if DRY:
    sid = sets.set_id.iloc[0]
    for j in ["Budget hunters", "Generic shopper"]:
        s, u = prompt(sid, j)
        print(f"\n######## {j} ########\n[SYSTEM]\n{s}\n\n[USER]\n{u}")
    raise SystemExit

from openai import OpenAI
client = OpenAI(timeout=120)

def call(system, user):
    kw = dict(model=MODEL, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
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

results, lock, done = [], threading.Lock(), set()
if RESUME and os.path.exists(OUT):
    prev = pd.read_parquet(OUT); results = prev.to_dict("records"); done = set(zip(prev.set_id, prev.judge))
    print(f"resuming: {len(done)} judgments already done")
MATCHED_ONLY = os.getenv("MATCHED_ONLY") == "1"   # only the shopper's own segment agent + generic judge
seg_of = sets.groupby("set_id").segment.first().to_dict()
tasks = [(sid, j) for sid in sets.set_id.unique() for j in judges
         if (sid, j) not in done and (not MATCHED_ONLY or j in ("Generic shopper", seg_of[sid]))]
print(f"{len(tasks)} judge calls with {WORKERS} workers")

def work(task):
    sid, jname = task
    g = groups[sid]
    got = {int(j["i"]): j.get("p_buy") for j in call(*prompt(sid, jname))}
    return [{"set_id": sid, "segment": g.segment.iloc[0], "judge": jname, "i": int(r.i),
             "parent_asin": r.parent_asin, "is_bought": bool(r.is_bought), "pop_A": r.pop_A,
             "p_buy": got.get(int(r.i))} for r in g.itertuples()]

t0, n_done, n_fail = time.time(), 0, 0
with ThreadPoolExecutor(WORKERS) as ex_:
    futs = {ex_.submit(work, t): t for t in tasks}
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
print(f"\nsaved {OUT} — run: python scripts/10_calibration_report.py {OUT}")
