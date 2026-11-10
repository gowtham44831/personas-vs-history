"""Persona profiles from segment statistics (frozen v2 rules).  CAT=Video_Games python scripts/04_profiles.py"""
import json, statistics
from cfg import KB, DEVICE_WORD, LEGACY

stats = json.load(open(f"{KB}/segment_stats.json"))
if LEGACY and "eco_shares" not in next(iter(stats.values())):
    raise SystemExit("phones profiles are frozen (v2, data/kb/profiles.json); not regenerating")
pop_mid = statistics.median(s["median_product_reviews"] for s in stats.values())
pct = lambda x: f"{round(x * 100)}%"

profiles = {}
for name, s in stats.items():
    bands = "; ".join(f"{leaf} ${b['p25']:.2f}-${b['p75']:.2f} (typically ${b['p50']:.2f})"
                      for leaf, b in s["price_bands_paid"].items() if leaf != "Unknown")
    subcats = ", ".join(k for k in s["top_subcategories"] if k != "Unknown")
    brands = ", ".join(list(s["top_brands"])[:5])
    eco = s["eco_shares"]
    eco_txt = ", ".join(f"{pct(v)} for {k}" for k, v in sorted(eco.items(), key=lambda x: -x[1]))
    facts = [
        f"You are a shopper in the '{name}' segment ({pct(s['share_of_pilot_users'])} of shoppers).",
        f"Price: {pct(s['budget_share'])} of your purchases are in the cheapest quarter of their "
        f"subcategory and {pct(s['premium_share'])} in the most expensive quarter.",
        f"Typical prices you pay: {bands}.",
        f"Platforms: of what you buy, {eco_txt}.",
        f"Brands: you buy across many brands (most often {brands}); "
        f"your favourite brand is about {pct(s['brand_loyalty'])} of your purchases.",
        f"You mostly buy: {subcats}."]
    cues = []
    budget, premium = s["budget_share"] > 0.4, s["premium_share"] > 0.25
    if budget:
        cues.append("You shop in a lower price band and avoid clearly overpriced items (more than about 2x your typical "
                    "price for that type of item). Within your price band you do not simply pick the cheapest option: "
                    "you choose well-reviewed, popular products.")
    if premium:
        cues.append("You favour established, higher-quality products; very cheap no-name items appeal less.")
    top_eco, top_share = max(eco.items(), key=lambda x: x[1])
    if top_share > 0.6:
        cues.append(f"Assume you own {top_eco} {DEVICE_WORD} unless the query names another brand or model; "
                    f"items for other {DEVICE_WORD} are irrelevant.")
    else:
        cues.append(f"You own {DEVICE_WORD} from more than one ecosystem; judge compatibility strictly by what the query names.")
    if not budget and not premium:
        cues.append("You buy often and are open to good alternatives (substitutes) if they fit your platform.")
    if s["median_product_reviews"] > pop_mid:
        cues.append("You prefer bestsellers with many reviews over little-known listings.")
    elif not budget:
        cues.append("You are comfortable buying lesser-known listings if the specs and price fit.")
    cues.append("An item that does not fit the platform or product type in the query is irrelevant regardless of price.")
    text = "\n".join(facts) + "\n\nHow you judge search results:\n" + "\n".join(f"- {c}" for c in cues)
    profiles[name] = {"segment_id": s["segment_id"], "profile": text}
    print(f"\n===== {name} =====\n{text}")
json.dump(profiles, open(f"{KB}/profiles.json", "w"), indent=2)
print(f"\nsaved {KB}/profiles.json")
