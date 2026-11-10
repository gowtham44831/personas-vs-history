# Personal History Beats Personas

Code for the paper **"Personal History Beats Personas: Calibrating LLM Shopper Judges Against Real Purchases"**.

The study asks how an LLM relevance judge should be told who the shopper is: through a behavioural
**segment persona**, through the shopper's **own purchase history**, or both. Judges are tested on
choice sets built from real, held-out Amazon purchases (the bought product plus 9 comparable products
the shopper did not buy), scored by the AUC of the bought item.

## Data (not included)

All data is public. Download it yourself; nothing from it is committed here.

| Source | What to download | Put it in |
|---|---|---|
| [Amazon Shopping Queries (ESCI)](https://github.com/amazon-science/esci-data) | `git clone` the repo (needs Git LFS) | `data/raw/esci-data/` |
| [Amazon Reviews 2023](https://amazon-reviews-2023.github.io) | `review` and `meta` files for `Cell_Phones_and_Accessories` and `Video_Games` | `data/raw/amazon2023/reviews/`, `data/raw/amazon2023/meta/` |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # then edit .env and paste your own API key
set -a && source .env && set +a   # load the settings into this terminal
```

The scripts read their settings from environment variables, so nothing secret is stored in the code:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI key (required for the judge) |
| `JUDGE_MODEL` | Judge model; the paper uses `gpt-4.1-mini` |
| `JUDGE_BASE_URL` | Optional: `http://localhost:11434/v1` to use a local Ollama model with `06b_judge.py` |

`.env` is git-ignored. Run `set -a && source .env && set +a` again in each new terminal, or `export` the
variables directly. On macOS, do not start the scripts through `caffeinate`; it breaks the OpenAI
client's SSL setup. Run `caffeinate -i` in a separate terminal instead.

Every script runs from the repository root. Select the category with `CAT=` (default
`Cell_Phones_and_Accessories`; also `Video_Games`). Category settings (ecosystem patterns,
query-type patterns, segment names) live in `config/<CAT>.json`; paths are handled by `scripts/cfg.py`.

## Pipeline

| Step | Script | Output |
|---|---|---|
| Prepare purchases, metadata, ESCI join | `01_prepare.py` | `data/processed/...` |
| Behavioural segments (k-means) | `02b_segments_v2.py` | user → segment table |
| Segment statistics | `03_segment_stats.py` | `data/kb/.../segment_stats.json` |
| Persona profiles (fixed rules) | `04_profiles.py` | `data/kb/.../profiles.json` |
| ESCI pilot queries | `05_pilot_queries.py` (`05b_fix_types.py` for phones) | query and candidate tables |
| Judge test on search results | `06b_judge.py`, report `07_judge_report.py`, order noise `08_noise.py` | Table 2 |
| Calibration gate (choice sets) | `09_calibration.py`, report `10_calibration_report.py` | Table 3, Figure 1 |
| Diagnostics | `11_diagnose.py` | Section 6.2 |
| Few-shot examples | `12_fewshot_calibration.py` | Section 6.2 |
| Rules from own history | `13_rules.py` | Table 4 |
| Own-history judges | `14_individual_agent.py`, report `15_individual_report.py` | Table 5, Figure 2 |
| Figures | `paper/figs.py` | Figures 1–2 |

`06_judge_test.py` and `02_segments.py` are earlier versions kept for reference.

### Reproducing the main results

```bash
# test window: personas vs generic vs mismatched (Table 3, Figure 1)
JUDGE_MODEL=gpt-4.1-mini N_SETS=200 WINDOW=test python scripts/09_calibration.py
python scripts/10_calibration_report.py data/results/calib_gpt_4_1_mini_test_n200.parquet
WINDOW=test N_SETS=200 python scripts/13_rules.py

# holdout (Jan-Sep 2023): own history vs personas (Table 5, Figure 2)
JUDGE_MODEL=gpt-4.1-mini N_SETS=200 WINDOW=holdout MATCHED_ONLY=1 python scripts/09_calibration.py
JUDGE_MODEL=gpt-4.1-mini N_SETS=200 WINDOW=holdout python scripts/14_individual_agent.py
WINDOW=holdout N_SETS=200 python scripts/15_individual_report.py
```

Prefix each command with `CAT=Video_Games` for the second category. Time windows: segments and
history 2019–2021; development Jan–Jun 2022; test Jul–Dec 2022; holdout Jan–Sep 2023.
A full category run is several thousand judge calls (a few dollars with gpt-4.1-mini).

## Citation

If you use this code, please cite the paper (preprint link to be added).

## License

MIT (code only). The Amazon datasets are subject to their own terms.
