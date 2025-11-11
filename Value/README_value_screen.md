# Value Investing Stock Screener (Buffett/Munger-leaning)

## Quick Start
1. Install dependencies:
   ```bash
   pip install pandas numpy requests
   ```
2. Put your tickers in `universe.csv` (one per line under `symbol`).
3. Get a FinancialModelingPrep API key (free tier works for basic endpoints).
4. Run:
   ```bash
   export FMP_API_KEY=YOUR_KEY
   python value_screen.py --mode fmp --input universe.csv --out results.csv
   ```
5. Open `results.csv` and `results_audit.txt`.

## What it does
- Pulls fundamentals & quotes
- Computes ROIC, FCF, leverage/coverage, accruals proxy, EV/EBIT & EV/FCF
- Performs a conservative two‑stage reverse DCF to get fair value per share
- Applies quality and valuation gates (defaults: Quality≥70, MOS≥30%)
- Ranks survivors by a 50/50 blend of quality and value scores

## Tuning
- `--wacc` (default 10%): raise if you want stricter required returns.
- `--g1` (default 4% for 5 yrs): lower for cyclicals; modest for wide-moat.
- `--gt` (default 2%): long‑run terminal growth.
- `--min-quality` and `--min-mos`: tighten/loosen to change hit rate.

## CSV Mode
If you prefer your own data, prepare `fundamentals.csv` with the fields listed in the script docstring.
Then run:
```bash
python value_screen.py --mode csv --input fundamentals.csv --out results.csv
```

## Notes
- Free FMP tier may lack some liquidity fields; you can merge ADV from your broker.
- For highly cyclical names, consider replacing EBIT with mid‑cycle averages.
- For financials (banks/insurers), ROIC and EV/EBIT are less meaningful; skip or use sector‑aware filters.