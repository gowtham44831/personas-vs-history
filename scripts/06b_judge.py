"""Judge test v2: schema-enforced output, saves after every query, resumable.
Ollama:  JUDGE_BASE_URL=http://localhost:11434/v1 JUDGE_MODEL=qwen3:8b python scripts/06b_judge.py
OpenAI:  JUDGE_MODEL=<model> python scripts/06b_judge.py      (needs OPENAI_API_KEY)
Options: N_QUERIES (default 8), MAX_CANDS (12), RESUME=1, JUDGE_TEMPERATURE (0 | none)
"""
import os, re, json, time, urllib.request
from cfg import PROC, KB, RES
import pandas as pd

BASE = os.getenv("JUDGE_BASE_URL", "")
MODEL = os.environ["JUDGE_MODEL"]
N = int(os.getenv("N_QUERIES", "8"))
MAXC = int(os.getenv("MAX_CANDS", "12"))
TEMP = os.getenv("JUDGE_TEMPERATURE", "0")
RESUME = os.getenv("RESUME") == "1"
SHUFFLE = int(os.getenv("SHUFFLE_SEED", "1"))
ONLY = [x.strip() for x in os.getenv("JUDGES", "").split(",") if x.strip()]
TAG = os.getenv("TAG", "")
OLLAMA = "11434" in BASE

SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["judgments"],
    "properties": {"judgments": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["i", "grade", "p_click"],
        "properties": {"i": {"type": "integer"},
                       "grade": {"type": "integer", "enum": [0, 1, 2, 3]},
                       "p_click": {"type": "number"}}}}}}

RUBRIC = """You judge e-commerce search results for one specific shopper.
Grade each product for the query from that shopper's point of view:
3 = exactly what this shopper wants and would consider buying
2 = acceptable substitute; they might buy it
1 = related, but they would not buy it
0 = irrelevant
Also give p_click: a number from 0 to 1 for how likely this shopper is to click it if shown.
Judge each product on its own merits; ignore the order in which products are listed.
Return one entry per product, using its number as "i"."""

if not OLLAMA:
    from openai import OpenAI
    client = OpenAI(timeout=300)

def call(system, user):
    t = time.time()
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if OLLAMA:
        body = {"model": MODEL, "stream": False, "format": SCHEMA, "messages": msgs,
                "options": {"temperature": 0, "num_ctx": 8192}}
        if re.search(r"qwen3|deepseek-r1|gpt-oss", MODEL):
            body["think"] = False
        req = urllib.request.Request(BASE.split("/v1")[0] + "/api/chat",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            out = json.load(r)
        txt, tin, tout = out["message"]["content"], out.get("prompt_eval_count"), out.get("eval_count")
    else:
        kw = dict(model=MODEL, messages=msgs, response_format={
            "type": "json_schema",
            "json_schema": {"name": "judgments", "strict": True, "schema": SCHEMA}})
        if TEMP != "none":
            kw["temperature"] = float(TEMP)
        try:
            r = client.chat.completions.create(**kw)
        except Exception as e:
            if "temperature" not in str(e) or "temperature" not in kw:
                raise
            kw.pop("temperature")
            r = client.chat.completions.create(**kw)
        txt, tin, tout = r.choices[0].message.content, r.usage.prompt_tokens, r.usage.completion_tokens
    return json.loads(txt)["judgments"], time.time() - t, tin, tout

def fmt(i, r):
    price = f"${r.price:.2f}" if pd.notna(r.price) else "price unknown"
    rating = (f"{r.average_rating:.1f} stars, {int(r.rating_number)} ratings"
              if pd.notna(r.average_rating) and pd.notna(r.rating_number) else "rating unknown")
    bullets = str(r.product_bullet_point or "")[:250].replace("\n", " ")
    return f"[{i}] {r.product_title} | brand: {r.product_brand or 'unknown'} | {price} | {rating} | {bullets}"

profiles = json.load(open(f"{KB}/profiles.json"))
judges = {"Generic shopper": "You are a typical online shopper."}
judges.update({k: v["profile"] for k, v in profiles.items()})
if ONLY:
    judges = {k: v for k, v in judges.items() if k in ONLY}
    assert judges, f"no judges match {ONLY}"

qs = pd.read_parquet(f"{PROC}/pilot_queries.parquet")
cands = pd.read_parquet(f"{PROC}/pilot_candidates.parquet")
MIX = os.getenv("Q_MIX")  # e.g. "generic=12,apple=4,samsung=4,other_brand=4"
if MIX:
    parts = [(k.strip(), int(v)) for k, v in (x.split("=") for x in MIX.split(","))]
    sample = pd.concat([qs[qs.query_type == k].sample(n=v, random_state=7) for k, v in parts])
else:
    sample = qs.groupby("query_type").sample(n=max(1, N // 4), random_state=7)

safe = re.sub(r"[^A-Za-z0-9]+", "_", MODEL) + (f"_{TAG}" if TAG else "")
out_rows, out_calls = f"{RES}/judge_{safe}.parquet", f"{RES}/judge_{safe}_calls.parquet"
rows, calls, done = [], [], set()
if RESUME and os.path.exists(out_rows):
    rows = pd.read_parquet(out_rows).to_dict("records")
    calls = pd.read_parquet(out_calls).to_dict("records")
    done = {r["query_id"] for r in rows}
    print(f"resuming: {len(done)} queries already judged")

for _, q in sample.iterrows():
    if q.query_id in done:
        continue
    c = cands[cands.query_id == q.query_id].sample(frac=1, random_state=1).head(MAXC)
    if SHUFFLE != 1:
        c = c.sample(frac=1, random_state=SHUFFLE)
    c = c.reset_index(drop=True)
    user = f"Search query: {q.query}\n\nProducts:\n" + "\n".join(fmt(i + 1, r) for i, r in c.iterrows())
    for jname, persona in judges.items():
        got, rec = {}, {"query_id": q.query_id, "judge": jname, "ok": False}
        for attempt in (1, 2):
            try:
                js, secs, tin, tout = call(persona + "\n\n" + RUBRIC, user)
                got = {int(j["i"]): j for j in js}
                rec.update(ok=True, secs=secs, tok_in=tin, tok_out=tout, attempt=attempt)
                break
            except Exception as e:
                print(f"  attempt {attempt} failed: {q.query!r} / {jname}: {str(e)[:150]}")
        calls.append(rec)
        for i, r in c.iterrows():
            j = got.get(i + 1, {})
            rows.append({"query_id": q.query_id, "query": q.query, "query_type": q.query_type,
                         "judge": jname, "product_id": r.product_id, "esci_label": r.esci_label,
                         "grade": j.get("grade"), "p_click": j.get("p_click")})
    pd.DataFrame(rows).to_parquet(out_rows, index=False)
    pd.DataFrame(calls).to_parquet(out_calls, index=False)
    ok = [x for x in calls if x["ok"] and x["query_id"] == q.query_id]
    avg = sum(x["secs"] for x in ok) / max(1, len(ok))
    print(f"done: {q.query} ({q.query_type}) | {len(ok)}/{len(judges)} ok | {avg:.0f}s per call")

print(f"\nsaved {out_rows} — run: python scripts/07_judge_report.py {out_rows}")
