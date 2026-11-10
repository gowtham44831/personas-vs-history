import os, re, json, time
import numpy as np, pandas as pd
from openai import OpenAI
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

BASE = os.getenv("JUDGE_BASE_URL")              # unset = OpenAI; Ollama = http://localhost:11434/v1
MODEL = os.environ["JUDGE_MODEL"]
N = int(os.getenv("N_QUERIES", "8"))
MAXC = int(os.getenv("MAX_CANDS", "12"))
TEMP = os.getenv("JUDGE_TEMPERATURE", "0")
client = OpenAI(base_url=BASE, api_key=os.getenv("OPENAI_API_KEY", "ollama"))

RUBRIC = """You judge e-commerce search results for one specific shopper.
Grade each product for the query from that shopper's point of view:
3 = exactly what this shopper wants and would consider buying
2 = acceptable substitute; they might buy it
1 = related, but they would not buy it
0 = irrelevant
Also estimate p_click: the probability (0 to 1) this shopper would click it if shown.
Judge each product on its own merits; ignore the order in which products are listed.
Return only JSON: {"judgments": [{"i": <product number>, "grade": <0-3>, "p_click": <0-1>}]}
with exactly one entry per product."""

profiles = json.load(open("data/kb/profiles.json"))
judges = {"Generic shopper": "You are a typical online shopper."}
judges.update({k: v["profile"] for k, v in profiles.items()})

qs = pd.read_parquet("data/processed/pilot_queries.parquet")
cands = pd.read_parquet("data/processed/pilot_candidates.parquet")
sample = qs.groupby("query_type").sample(n=max(1, N // 4), random_state=7)

def fmt(i, r):
    price = f"${r.price:.2f}" if pd.notna(r.price) else "price unknown"
    rating = (f"{r.average_rating:.1f} stars, {int(r.rating_number)} ratings"
              if pd.notna(r.average_rating) and pd.notna(r.rating_number) else "rating unknown")
    bullets = str(r.product_bullet_point or "")[:250].replace("\n", " ")
    return f"[{i}] {r.product_title} | brand: {r.product_brand or 'unknown'} | {price} | {rating} | {bullets}"

def call(system, user):
    kw = dict(model=MODEL, messages=[{"role": "system", "content": system},
                                     {"role": "user", "content": user}],
              response_format={"type": "json_object"})
    if TEMP != "none":
        kw["temperature"] = float(TEMP)
    t = time.time()
    r = client.chat.completions.create(**kw)
    txt = re.sub(r"<think>.*?</think>", "", r.choices[0].message.content or "", flags=re.S)
    m = re.search(r"[\[{].*[\]}]", txt, re.S)
    data = json.loads(m.group(0))
    if isinstance(data, dict):
        lists = [v for v in data.values() if isinstance(v, list)]
        if not lists:
            raise ValueError(f"no list in reply: {txt[:200]}")
        data = lists[0]
    u = r.usage
    return data, time.time() - t, (u.prompt_tokens if u else None), (u.completion_tokens if u else None)

rows, calls = [], []
for _, q in sample.iterrows():
    c = cands[cands.query_id == q.query_id].sample(frac=1, random_state=1).head(MAXC).reset_index(drop=True)
    products = "\n".join(fmt(i + 1, r) for i, r in c.iterrows())
    user = f"Search query: {q.query}\n\nProducts:\n{products}"
    if "qwen3" in MODEL:
        user += "\n/no_think"
    for jname, persona in judges.items():
        try:
            js, secs, tin, tout = call(persona + "\n\n" + RUBRIC, user)
            calls.append({"judge": jname, "secs": secs, "tok_in": tin, "tok_out": tout, "ok": True})
            got = {}
            for pos, j in enumerate(js, 1):
                if not isinstance(j, dict):
                    continue
                key = next((j[k] for k in ("i", "id", "index", "product", "product_number", "number") if k in j), pos)
                digits = re.sub(r"\D", "", str(key))
                got[int(digits) if digits else pos] = j
        except Exception as e:
            print(f"  failed: {q.query!r} / {jname}: {e}")
            calls.append({"judge": jname, "ok": False}); got = {}
        for i, r in c.iterrows():
            j = got.get(i + 1, {})
            rows.append({"query_id": q.query_id, "query": q.query, "query_type": q.query_type,
                         "judge": jname, "product_id": r.product_id, "esci_label": r.esci_label,
                         "grade": j.get("grade", j.get("score", j.get("relevance"))), "p_click": j.get("p_click", j.get("click_probability"))})
    print(f"done: {q.query} ({q.query_type})")

res = pd.DataFrame(rows); cl = pd.DataFrame(calls)
safe = re.sub(r"[^A-Za-z0-9]+", "_", MODEL)
res.to_parquet(f"data/results/judge_test_{safe}.parquet", index=False)

print("\n=== speed ===")
ok = cl[cl.ok]
print(f"calls ok: {len(ok)}/{len(cl)} | sec/call: {ok.secs.mean():.1f} | tokens in/out per call: "
      f"{ok.tok_in.mean():.0f}/{ok.tok_out.mean():.0f}")

print("\n=== agreement with ESCI human labels (E=3, S=2, C=1, I=0) ===")
res["esci_grade"] = res.esci_label.map({"E": 3, "S": 2, "C": 1, "I": 0})
for jname, g in res.dropna(subset=["grade"]).groupby("judge"):
    g = g.astype({"grade": int})
    k = cohen_kappa_score(g.esci_grade, g.grade, weights="quadratic")
    rho = spearmanr(g.esci_grade, g.grade).statistic
    print(f"{jname:30s} weighted kappa={k:.2f}  spearman={rho:.2f}  n={len(g)}")

print("\n=== do segments disagree? ===")
seg = res[res.judge != "Generic shopper"].dropna(subset=["grade"])
spread = seg.pivot_table(index=["query_id", "product_id"], columns="judge", values="grade").dropna()
rng = spread.max(axis=1) - spread.min(axis=1)
print(f"products judged by all 4 segments: {len(spread)}")
print(f"segments differ by >=1 grade: {(rng >= 1).mean():.0%} | by >=2 grades: {(rng >= 2).mean():.0%}")
print("\nmean grade by judge and query type:")
print(res.dropna(subset=["grade"]).pivot_table(index="judge", columns="query_type",
      values="grade", aggfunc="mean").round(2))
