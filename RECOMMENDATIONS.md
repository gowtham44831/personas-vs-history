# Recommendations for practitioners

**In short:** when an LLM judges search relevance, tell it who the shopper is with **their own recent
history**, not a segment persona, and **check the judge against real customer behaviour before trusting it.**

These recommendations follow from the results in the paper (phone accessories and video games,
public Amazon data, gpt-4.1-mini as judge). Check them on your own categories before relying on them.

## How to check and adopt this

1. **Build a test set from your own data.** Take recent real purchases (or clicks / add-to-carts) and make
   choice sets: the item the customer chose plus about 9 comparable items they did not choose. A few hundred
   sets per segment is enough. With search logs, use real results pages: shown, clicked and skipped.
2. **Measure the generic judge first.** Run the LLM as "a typical shopper" and compute the AUC of the chosen
   item (`scripts/09_calibration.py` and `scripts/10_calibration_report.py` show how). This is your baseline,
   and it often reveals segments the generic judge serves badly.
3. **Add each shopper's own history to the prompt.** Keep it short: which devices, platforms or sizes they buy
   for, and their last 5–10 purchases or clicks; for search, their recent queries
   (see `scripts/14_individual_agent.py`). Compare against the baseline on the same sets.
4. **Test on a later time window before relying on it.** Fix the prompt on older data, then check once on
   newer data the judge has not seen. Adopt it only if the gain holds.
5. **Use segments for reporting, not as personas.** Sample real customers from each segment, judge each with
   their own history, and average by segment. This gives per-segment scorecards showing which groups a search
   change helps or hurts.
6. **Put it to work.** Score candidate ranking models offline, flag irrelevant products per query, and rerun
   on every release as a regression check. Confirm big decisions with an A/B test.

## What not to do

- **Don't build personas around price.** "Budget" shoppers did not pick the cheapest items; they picked
  well-reviewed ones that fit their device. "Cheapest first" scored below random in both categories.
- **Don't stack a persona on top of individual history.** It added nothing and sometimes hurt.
- **Don't expect example purchases in the prompt to help.** Few-shot examples did not improve the judge.
- **Don't assume one category's result transfers.** History helped much more for phone accessories
  (device fit) than for video games (content taste).

## Practical cautions

- **New customers with no history:** fall back to a persona if their segment is homogeneous (for example,
  single-platform), otherwise to the generic judge.
- **Privacy:** send the LLM only what is needed (a device summary and recent item titles), with no names,
  addresses or IDs, and use an approved model provider or a self-hosted model.
- **Keep it calibrated:** rerun the check periodically; catalogs, customers and model versions change.
- **Cost is low:** about 2 seconds and a fraction of a cent per judgment with a small model, so a full
  check costs a few dollars to tens of dollars.

## Where it could go further

Companies with **search and click logs** can likely do better than this public-data study: recent queries
and clicks show what a shopper wants *now*, which purchase history only approximates. Testing
search-history conditioning is the most promising next step.
