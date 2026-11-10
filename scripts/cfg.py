"""Category configuration shared by all scripts. Select with CAT=<Amazon category>, e.g. CAT=Video_Games.
Phones (the first pilot) keep their original paths; other categories get their own sub-folders."""
import os, re, json

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAT = os.getenv("CAT", "Cell_Phones_and_Accessories")
C = json.load(open(os.path.join(_root, "config", f"{CAT}.json")))
SLUG = C["slug"]
LEGACY = SLUG == "phones"
RAW = "data/raw"
PROC = "data/processed" if LEGACY else f"data/processed/{SLUG}"
KB = "data/kb" if LEGACY else f"data/kb/{SLUG}"
RES = "data/results" if LEGACY else f"data/results/{SLUG}"
ECO = C["ecosystems"]                 # name -> regex on product titles
QTYPES = C["query_types"]             # [name, regex] applied in order; later matches override
SEG_NAMES = {int(k): v for k, v in C.get("segment_names", {}).items()}
DEVICE_WORD = C.get("device_word", "devices")
for _d in (PROC, KB, RES):
    os.makedirs(_d, exist_ok=True)

def eco_of(title):
    t = str(title).lower()
    for name, pat in ECO.items():
        if re.search(pat, t):
            return name
    return "Other"

def eco_col(name):
    return "share_" + re.sub(r"[^a-z0-9]+", "_", name.lower())
