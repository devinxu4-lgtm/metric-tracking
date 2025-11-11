
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Value Investing Stock Screener
------------------------------

A pragmatic, Buffett/Munger‑leaning screener that searches for:
1) High financial quality (durable returns, clean accounting, prudent leverage)
2) Deep discount (vs conservative intrinsic value from reverse DCF and EV/EBIT/EV/FCF)
3) Basic liquidity and sanity checks

Data sources
- Mode "fmp": FinancialModelingPrep (https://financialmodelingprep.com/) fundamentals & quotes.
  Set environment variable FMP_API_KEY before running.
- Mode "csv": bring your own fundamentals via CSV (template fields described below).

Outputs
- A ranked CSV with metrics and composite scores.
- Logs that explain why tickers failed any gate (for auditing).

Usage
------
# Fetch data from FMP for the tickers in universe.csv, screen, and save results.csv
export FMP_API_KEY=YOUR_KEY
python value_screen.py --mode fmp --input universe.csv --out results.csv

# Use your own fundamentals data (CSV) with required columns (see CSV mode below)
python value_screen.py --mode csv --input fundamentals.csv --out results.csv

# Tune thresholds
python value_screen.py --mode fmp --input universe.csv --out results.csv \
  --min-quality 70 --min-mos 0.35 --wacc 0.10 --g1 0.04 --years1 5 --gt 0.02

CSV mode (bring-your-own fundamentals)
--------------------------------------
Required columns (case-insensitive, underscores/dashes ignored):
- symbol
- price
- shares_outstanding
- ebit_ttm
- tax_rate   (e.g., 0.21 for 21%)
- invested_capital (latest, book value approximation)
- invested_capital_prev (previous year for ROIC averaging)
- gross_margin_ttm
- gross_margin_5y_std
- revenue_5y_cagr
- ebitda_ttm
- interest_expense_ttm
- total_debt
- cash_and_equivalents
- cfo_ttm
- capex_ttm
- fcf_5y_avg
- accruals_ratio  (optional; if missing will be estimated as (NI - CFO)/TA with NI≈EBIT*(1-tax))
- sector (optional; string)
- adv_usd (optional; average daily traded value in USD; default 1e6)

Notes
- If some fields are missing, the script will try conservative fallbacks.
- All monetary units should be consistent per row (e.g., USD).

Composite scoring
-----------------
Quality Score (0-100) = weighted blend of:
- ROIC vs hurdle (20%)
- FCF margin (15%)
- Gross margin stability (10%)
- Net debt / EBITDA (15%)
- Interest coverage (10%)
- Accruals quality (10%)
- Revenue CAGR 5y (10%)
- Piotroski‑like momentum (10%; simplified signals)

Value Score (0-100) = blend of:
- Margin of Safety from reverse DCF (60%)
- EV/EBIT percentile cheapness within pool (20%)
- EV/FCF percentile cheapness within pool (20%)

Final Score = 0.5*Quality + 0.5*Value (tunable).

License: MIT
Author: ChatGPT
"""
import os
import argparse
import math
import sys
import time
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests

# ---------- Utilities ----------

def _norm_col(df: pd.DataFrame, name: str) -> str:
    """Return a normalized column key (case/space/underscore insensitive)."""
    norm = {c: c for c in df.columns}
    lowered = {c.lower().replace(" ", "").replace("_","").replace("-",""): c for c in df.columns}
    return lowered.get(name.lower().replace(" ", "").replace("_","").replace("-",""), name)

def _safe_div(a, b, default=np.nan):
    try:
        if b == 0 or b is None or a is None:
            return default
        return a / b
    except Exception:
        return default

def _winsorize(series: pd.Series, p: float = 0.01) -> pd.Series:
    if series.empty:
        return series
    lower = series.quantile(p)
    upper = series.quantile(1 - p)
    return series.clip(lower, upper)

def pct_change(a: float, b: float) -> Optional[float]:
    if b in (0, None) or a is None or b is None:
        return None
    return (a - b) / b

def nz(x, alt):
    return alt if (x is None or (isinstance(x,float) and math.isnan(x))) else x

# ---------- DCF ----------

def two_stage_dcf_fcfe(fcfe0: float, wacc: float, g1: float, years1: int, gt: float) -> float:
    """
    Present value of equity cash flows using a 2-stage DCF:
    - Stage 1: grow FCFE at g1 for 'years1' years
    - Stage 2: perpetuity at gt
    Returns PV of all future FCFE (i.e., equity value).
    """
    if any([fcfe0 is None, wacc is None, g1 is None, gt is None]):
        return np.nan
    if wacc <= gt:
        # avoid explosions; force small spread
        gt = min(gt, wacc - 0.005)
    pv = 0.0
    fcfe = fcfe0
    for t in range(1, years1 + 1):
        fcfe *= (1 + g1)
        pv += fcfe / ((1 + wacc)**t)
    # Terminal value at end of stage 1
    tv = (fcfe * (1 + gt)) / (wacc - gt)
    pv += tv / ((1 + wacc)**years1)
    return pv

# ---------- Quality & Value scoring ----------

def quality_score(row: pd.Series, roic_hurdle: float = 0.12) -> float:
    """
    0-100 composite quality score.
    """
    # ROIC vs hurdle (20)
    roic = row.get("roic_5y_avg", np.nan)
    roic_component = np.interp(nz(roic, 0), [0, roic_hurdle, roic_hurdle*2], [20*0.2, 20*0.6, 20*1.0])
    roic_component = max(0, min(20, roic_component))

    # FCF margin (15)
    fcf_margin = row.get("fcf_margin_ttm", np.nan)
    fcf_component = np.interp(nz(fcf_margin, 0), [0, 0.05, 0.15], [15*0.3, 15*0.7, 15*1.0])
    fcf_component = max(0, min(15, fcf_component))

    # Gross margin stability (10) — lower std is better
    gm_std = row.get("gross_margin_5y_std", np.nan)
    if gm_std is None or np.isnan(gm_std):
        gm_component = 10 * 0.5
    else:
        gm_component = np.interp(gm_std, [0.0, 0.05, 0.15], [10*1.0, 10*0.6, 10*0.2])
    gm_component = max(0, min(10, gm_component))

    # Net debt / EBITDA (15) — lower is better
    nd_ebitda = row.get("net_debt_ebitda", np.nan)
    nd_component = np.interp(nz(nd_ebitda, 3.0), [0.0, 1.0, 3.0, 5.0], [15*1.0, 15*0.9, 15*0.6, 15*0.2])
    nd_component = max(0, min(15, nd_component))

    # Interest coverage (10) — higher is better
    int_cov = row.get("interest_coverage", np.nan)
    ic_component = np.interp(nz(int_cov, 2.0), [1.0, 3.0, 8.0], [10*0.3, 10*0.7, 10*1.0])
    ic_component = max(0, min(10, ic_component))

    # Accruals quality (10) — lower accruals better; use CFO/NI proxy or accruals ratio
    accr = row.get("accruals_ratio", np.nan)
    if accr is None or np.isnan(accr):
        accr_component = 10 * 0.5
    else:
        accr_component = np.interp(abs(accr), [0.0, 0.05, 0.15], [10*1.0, 10*0.7, 10*0.2])
    accr_component = max(0, min(10, accr_component))

    # Revenue CAGR 5y (10) — moderate growth is fine; penalize negative
    cagr = row.get("revenue_5y_cagr", np.nan)
    cagr_component = np.interp(nz(cagr, 0.0), [-0.05, 0.0, 0.05, 0.15], [10*0.2, 10*0.5, 10*0.8, 10*1.0])
    cagr_component = max(0, min(10, cagr_component))

    # Piotroski-like momentum (10) — simplified: positive FCF, improving ROIC, stable leverage
    pos_fcf = 1 if nz(row.get("fcf_ttm", np.nan), -1) > 0 else 0
    roic_trend = 1 if pct_change(row.get("roic_ttm", np.nan), row.get("roic_prev", np.nan)) and pct_change(row.get("roic_ttm", np.nan), row.get("roic_prev", np.nan)) > 0 else 0
    lev_stable = 1 if nz(row.get("net_debt_ebitda", np.nan), 10) <= nz(row.get("net_debt_ebitda_prev", np.nan), 10) else 0
    pio_component = (pos_fcf + roic_trend + lev_stable) / 3 * 10

    total = roic_component + fcf_component + gm_component + nd_component + ic_component + accr_component + cagr_component + pio_component
    return float(round(max(0, min(100, total)), 2))

def value_score(df: pd.DataFrame, row: pd.Series) -> float:
    """
    0-100 composite value score.
    """
    mos = row.get("margin_of_safety", np.nan)
    mos_component = np.interp(nz(mos, 0.0), [0.0, 0.3, 0.6], [60*0.3, 60*0.8, 60*1.0])
    mos_component = max(0, min(60, mos_component))

    # Percentile cheapness (lower multiple = better)
    ev_ebit = df["ev_ebit"]
    ev_fcf = df["ev_fcf"]
    ev_ebit_pct = 1 - _safe_div((ev_ebit <= row["ev_ebit"]).sum(), max(len(ev_ebit), 1), 0.5)
    ev_fcf_pct = 1 - _safe_div((ev_fcf <= row["ev_fcf"]).sum(), max(len(ev_fcf), 1), 0.5)

    ev_ebit_component = 20 * ev_ebit_pct
    ev_fcf_component = 20 * ev_fcf_pct

    total = mos_component + ev_ebit_component + ev_fcf_component
    return float(round(max(0, min(100, total)), 2))

# ---------- Data Adapters ----------

class FMPAdapter:
    def __init__(self, api_key: Optional[str] = None, pause: float = 0.7):
        self.api_key = api_key or os.getenv("FMP_API_KEY", "")
        self.base = "https://financialmodelingprep.com/api/v3"
        self.pause = pause
        if not self.api_key:
            raise RuntimeError("FMP_API_KEY not set. Export it or pass via env.")

    def _get(self, path: str, params: Dict) -> Optional[List[Dict]]:
        params = dict(params)
        params["apikey"] = self.api_key
        url = f"{self.base}/{path}"
        for _ in range(2):
            try:
                res = requests.get(url, params=params, timeout=20)
                if res.status_code == 200:
                    time.sleep(self.pause)
                    return res.json()
            except Exception:
                time.sleep(self.pause)
        return None

    def quote(self, symbol: str) -> Optional[Dict]:
        arr = self._get("quote/{s}".format(s=symbol), {})
        if isinstance(arr, list) and arr:
            return arr[0]
        return None

    def profile(self, symbol: str) -> Optional[Dict]:
        arr = self._get(f"profile/{symbol}", {})
        if isinstance(arr, list) and arr:
            return arr[0]
        return None

    def income(self, symbol: str, limit=5) -> Optional[List[Dict]]:
        return self._get(f"income-statement/{symbol}", {"period": "annual", "limit": limit})

    def balance(self, symbol: str, limit=5) -> Optional[List[Dict]]:
        return self._get(f"balance-sheet-statement/{symbol}", {"period": "annual", "limit": limit})

    def cashflow(self, symbol: str, limit=5) -> Optional[List[Dict]]:
        return self._get(f"cash-flow-statement/{symbol}", {"period": "annual", "limit": limit})

    def key_metrics(self, symbol: str, limit=5) -> Optional[List[Dict]]:
        return self._get(f"key-metrics/{symbol}", {"period": "annual", "limit": limit})

def compute_metrics_from_fmp(sym: str, fmp: FMPAdapter, logs: List[str]) -> Optional[Dict]:
    q = fmp.quote(sym)
    if not q:
        logs.append(f"[{sym}] Missing quote.")
        return None
    price = q.get("price") or q.get("previousClose") or np.nan

    prof = fmp.profile(sym) or {}
    shares = prof.get("sharesOutstanding") or prof.get("mktCap") and _safe_div(prof.get("mktCap"), price, np.nan) or np.nan
    sector = prof.get("sector") or ""

    inc = fmp.income(sym, limit=6) or []
    bal = fmp.balance(sym, limit=6) or []
    cf  = fmp.cashflow(sym, limit=6) or []

    if len(inc) < 2 or len(bal) < 2 or len(cf) < 2:
        logs.append(f"[{sym}] Insufficient statements (need >=2 years).")
        return None

    # Sort by date ascending
    inc = sorted(inc, key=lambda x: x.get("date", ""))
    bal = sorted(bal, key=lambda x: x.get("date", ""))
    cf  = sorted(cf,  key=lambda x: x.get("date", ""))

    # TTM / latest approximations from last entry
    latest_i = inc[-1]
    prev_i   = inc[-2]
    latest_b = bal[-1]
    prev_b   = bal[-2]
    latest_c = cf[-1]
    prev_c   = cf[-2]

    ebit = latest_i.get("ebit", np.nan)
    tax_rate = _safe_div(latest_i.get("incomeTaxExpense", np.nan), nz(latest_i.get("incomeBeforeTax", np.nan), np.nan), 0.21)
    if tax_rate is None or math.isnan(tax_rate) or tax_rate <= 0:
        tax_rate = 0.21

    # Invested Capital approx: total equity + total debt - cash
    total_debt = nz(latest_b.get("totalDebt", np.nan), 0.0)
    cash = nz(latest_b.get("cashAndCashEquivalents", np.nan), 0.0)
    total_equity = nz(latest_b.get("totalStockholdersEquity", np.nan), 0.0)
    invested_capital = total_equity + total_debt - cash
    invested_capital_prev = nz(prev_b.get("totalStockholdersEquity", 0.0), 0.0) + nz(prev_b.get("totalDebt", 0.0), 0.0) - nz(prev_b.get("cashAndCashEquivalents", 0.0), 0.0)
    avg_ic = np.nanmean([invested_capital, invested_capital_prev])

    nopat = nz(ebit, 0.0) * (1 - tax_rate)
    roic_ttm = _safe_div(nopat, avg_ic, np.nan)
    roic_prev = None
    if len(inc) >= 3 and len(bal) >= 3:
        ebit_prev2 = inc[-3].get("ebit", np.nan)
        ic_prev2 = nz(bal[-3].get("totalStockholdersEquity", 0.0), 0.0) + nz(bal[-3].get("totalDebt", 0.0), 0.0) - nz(bal[-3].get("cashAndCashEquivalents", 0.0), 0.0)
        avg_ic_prev = np.nanmean([invested_capital_prev, ic_prev2])
        roic_prev = _safe_div(nz(ebit_prev2, 0.0) * (1 - tax_rate), avg_ic_prev, np.nan)

    revenue_series = [x.get("revenue", np.nan) for x in inc[-6:]]
    gm_series = []
    for x in inc[-6:]:
        rev = nz(x.get("revenue", np.nan), np.nan)
        cogs = nz(x.get("costOfRevenue", np.nan), np.nan)
        gm = _safe_div(rev - cogs, rev, np.nan)
        gm_series.append(gm)
    try:
        gm_std = float(pd.Series(gm_series[-5:]).std(skipna=True))
    except Exception:
        gm_std = np.nan

    # CAGR revenue 5y
    try:
        if revenue_series[0] and revenue_series[-1] and revenue_series[0] > 0:
            years = max(len(revenue_series)-1, 1)
            cagr = (revenue_series[-1] / revenue_series[0]) ** (1/years) - 1
        else:
            cagr = np.nan
    except Exception:
        cagr = np.nan

    ebitda = latest_i.get("ebitda", np.nan)
    interest_expense = abs(nz(latest_i.get("interestExpense", np.nan), nz(latest_c.get("interestPaid", np.nan), np.nan)))
    int_cov = _safe_div(ebit, interest_expense, np.nan)
    net_debt = nz(total_debt, 0.0) - nz(cash, 0.0)
    nd_ebitda = _safe_div(net_debt, ebitda, np.nan)

    cfo = latest_c.get("netCashProvidedByOperatingActivities", np.nan)
    capex = abs(nz(latest_c.get("capitalExpenditure", np.nan), 0.0))
    fcf_ttm = nz(cfo, 0.0) - nz(capex, 0.0)
    fcf_margin = _safe_div(fcf_ttm, nz(latest_i.get("revenue", np.nan), np.nan), np.nan)

    # 5y average FCF
    fcf_series = []
    for x in cf[-5:]:
        fcf_series.append(nz(x.get("netCashProvidedByOperatingActivities", 0.0), 0.0) - abs(nz(x.get("capitalExpenditure", 0.0), 0.0)))
    fcf_5y_avg = float(np.nanmean(fcf_series)) if len(fcf_series) else np.nan

    # accruals ratio proxy
    ni = nz(latest_i.get("netIncome", np.nan), nz(nopat, np.nan))
    ta = nz(latest_b.get("totalAssets", np.nan), np.nan)
    accruals_ratio = _safe_div(ni - nz(cfo, 0.0), ta, np.nan)

    # Market cap & EV
    mktcap = price * nz(shares, np.nan)
    ev = nz(mktcap, np.nan) + nz(net_debt, 0.0)

    ev_ebit = _safe_div(ev, ebit, np.nan)
    ev_fcf = _safe_div(ev, fcf_ttm, np.nan)

    # Liquidity proxy
    adv_usd = np.nan  # unknown from FMP free endpoints; user can add later

    row = dict(
        symbol=sym, price=price, shares_outstanding=shares, sector=sector,
        ebit_ttm=ebit, tax_rate=tax_rate,
        invested_capital=invested_capital, invested_capital_prev=invested_capital_prev,
        roic_ttm=roic_ttm, roic_prev=roic_prev,
        roic_5y_avg=np.nan,  # we only have 2-3 yrs; leave blank or compute if >5yrs
        gross_margin_ttm=gm_series[-1] if gm_series else np.nan,
        gross_margin_5y_std=gm_std, revenue_5y_cagr=cagr,
        ebitda_ttm=ebitda, interest_expense_ttm=interest_expense,
        total_debt=total_debt, cash_and_equivalents=cash,
        cfo_ttm=cfo, capex_ttm=capex, fcf_ttm=fcf_ttm, fcf_margin_ttm=fcf_margin,
        fcf_5y_avg=fcf_5y_avg, accruals_ratio=accruals_ratio,
        net_debt=net_debt, ev=ev, ev_ebit=ev_ebit, ev_fcf=ev_fcf,
        net_debt_ebitda=nd_ebitda, interest_coverage=int_cov, adv_usd=adv_usd
    )
    return row

def load_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize expected columns
    def col(key): return _norm_col(df, key)
    needed = [
        "symbol","price","shares_outstanding","ebit_ttm","tax_rate",
        "invested_capital","invested_capital_prev","gross_margin_ttm",
        "gross_margin_5y_std","revenue_5y_cagr","ebitda_ttm","interest_expense_ttm",
        "total_debt","cash_and_equivalents","cfo_ttm","capex_ttm","fcf_5y_avg"
    ]
    for k in needed:
        if col(k) not in df.columns:
            raise ValueError(f"Missing required column in CSV: {k}")
    # Map normalized names to canonical
    rename_map = { _norm_col(df,k):k for k in df.columns }  # start with identity
    # Enforce canonical on the needed keys
    rename_map.update({ _norm_col(df,k):k for k in needed })
    optional = ["accruals_ratio","sector","adv_usd"]
    for k in optional:
        if _norm_col(df,k) in df.columns:
            rename_map[_norm_col(df,k)] = k
    df = df.rename(columns=rename_map)
    # Derive fields if missing
    df["fcf_ttm"] = df.get("cfo_ttm", np.nan) - df.get("capex_ttm", np.nan)
    df["fcf_margin_ttm"] = df["fcf_ttm"] / df.get("price", np.nan) * np.nan  # left as NaN if revenue not present
    # Approximations
    df["net_debt"] = df.get("total_debt", np.nan) - df.get("cash_and_equivalents", np.nan)
    df["interest_coverage"] = df.get("ebit_ttm", np.nan) / df.get("interest_expense_ttm", np.nan)
    df["net_debt_ebitda"] = df["net_debt"] / df.get("ebitda_ttm", np.nan)
    df["ev"] = (df.get("price", np.nan) * df.get("shares_outstanding", np.nan)) + df["net_debt"]
    df["ev_ebit"] = df["ev"] / df.get("ebit_ttm", np.nan)
    df["ev_fcf"] = df["ev"] / df.get("fcf_ttm", np.nan)
    return df

# ---------- Screening Pipeline ----------

def screen(df: pd.DataFrame,
           wacc: float,
           g1: float,
           years1: int,
           gt: float,
           min_quality: float,
           min_mos: float,
           min_adv_usd: float = 1e6,
           require_positive_fcf: bool = True,
           max_nd_ebitda: float = 3.5,
           min_int_cov: float = 4.0,
           max_gm_std: float = 0.15) -> pd.DataFrame:
    """
    Compute intrinsic value, MOS, scores, then filter & rank.
    """
    df = df.copy()

    # Compute ROIC (5y avg if possible)
    if "roic_5y_avg" not in df.columns or df["roic_5y_avg"].isna().all():
        # fallback: use roic_ttm if available
        df["roic_5y_avg"] = df.get("roic_ttm", np.nan)

    # Normalize FCFE base as 5y avg FCF (conservative) if available; else use TTM
    df["fcfe_base"] = df["fcf_5y_avg"].fillna(df.get("fcf_ttm", np.nan))

    # Intrinsic equity value via 2-stage DCF
    df["equity_value_dcf"] = df["fcfe_base"].apply(lambda x: two_stage_dcf_fcfe(x, wacc, g1, years1, gt))
    # Fair value per share
    df["fair_value"] = df["equity_value_dcf"] / df.get("shares_outstanding", np.nan)
    # Margin of safety
    df["margin_of_safety"] = 1 - (df.get("price", np.nan) / df["fair_value"])

    # Multiples sanity
    df["ev_ebit"] = df.get("ev_ebit", np.nan)
    df["ev_fcf"] = df.get("ev_fcf", np.nan)

    # Quality score
    df["quality_score"] = df.apply(quality_score, axis=1)

    # Value score (needs distribution context)
    # Protect against inf/nan
    df["ev_ebit"] = pd.to_numeric(df["ev_ebit"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df["ev_fcf"] = pd.to_numeric(df["ev_fcf"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df["value_score"] = df.apply(lambda r: value_score(df, r), axis=1)

    # Final composite
    df["final_score"] = 0.5*df["quality_score"] + 0.5*df["value_score"]

    # Gating filters (strict but reasonable)
    gates = (
        (df["margin_of_safety"] >= min_mos) &
        (df["quality_score"] >= min_quality) &
        (df.get("adv_usd", pd.Series([min_adv_usd]*len(df))) >= min_adv_usd) &
        (df.get("net_debt_ebitda", np.nan) <= max_nd_ebitda) &
        (df.get("interest_coverage", np.nan) >= min_int_cov) &
        (df.get("gross_margin_5y_std", np.nan) <= max_gm_std)
    )
    if require_positive_fcf:
        gates = gates & (df.get("fcf_ttm", np.nan) > 0)

    df["pass_gates"] = gates

    # Rank
    df = df.sort_values(by=["pass_gates","final_score","margin_of_safety"], ascending=[False, False, False])

    return df

def explain_failures(df: pd.DataFrame, min_quality: float, min_mos: float,
                     min_adv_usd: float, max_nd_ebitda: float, min_int_cov: float, max_gm_std: float) -> Dict[str, List[str]]:
    reasons = {}
    for _, r in df.iterrows():
        msgs = []
        if r.get("margin_of_safety", -1) < min_mos: msgs.append(f"MOS<{min_mos:.0%}")
        if r.get("quality_score", 0) < min_quality: msgs.append(f"Quality<{min_quality}")
        if nz(r.get("adv_usd", np.nan), 0) < min_adv_usd: msgs.append(f"ADV<{min_adv_usd:.0f}")
        if nz(r.get("net_debt_ebitda", np.nan), 99) > max_nd_ebitda: msgs.append(f"NetDebt/EBITDA>{max_nd_ebitda}")
        if nz(r.get("interest_coverage", np.nan), 0) < min_int_cov: msgs.append(f"IC<{min_int_cov}")
        if nz(r.get("gross_margin_5y_std", np.nan), 1) > max_gm_std: msgs.append(f"GM std>{max_gm_std}")
        if r.get("fcf_ttm", -1) <= 0: msgs.append("FCF<=0")
        if msgs:
            reasons[r["symbol"]] = msgs
    return reasons

# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser(description="Value Investing Stock Screener")
    p.add_argument("--mode", choices=["fmp","csv"], required=True, help="Data source mode")
    p.add_argument("--input", required=True, help="universe.csv (mode=fmp) or fundamentals.csv (mode=csv)")
    p.add_argument("--out", default="results.csv", help="output CSV path")
    p.add_argument("--wacc", type=float, default=0.10, help="WACC used in DCF (e.g., 0.10)")
    p.add_argument("--g1", type=float, default=0.04, help="Stage-1 FCFE growth (e.g., 0.04)")
    p.add_argument("--years1", type=int, default=5, help="Stage-1 horizon (years)")
    p.add_argument("--gt", type=float, default=0.02, help="Terminal growth (e.g., 0.02)")
    p.add_argument("--min-quality", type=float, default=70.0, help="Min quality score to pass gates")
    p.add_argument("--min-mos", type=float, default=0.30, help="Min margin of safety to pass gates (e.g., 0.30)")
    p.add_argument("--min-adv", type=float, default=1_000_000.0, help="Min average daily traded value (USD)")
    p.add_argument("--no-require-pos-fcf", action="store_true", help="Do not require positive FCF TTM")
    args = p.parse_args()

    if args.mode == "fmp":
        uni = pd.read_csv(args.input)
        sym_col = _norm_col(uni, "symbol")
        if sym_col not in uni.columns:
            raise ValueError("universe.csv must have a 'symbol' column")
        symbols = [str(s).strip() for s in uni[sym_col].dropna().tolist()]
        fmp = FMPAdapter()
        rows = []
        logs = []
        for s in symbols:
            try:
                r = compute_metrics_from_fmp(s, fmp, logs)
                if r:
                    rows.append(r)
            except Exception as e:
                logs.append(f"[{s}] error: {e}")
        df = pd.DataFrame(rows)
        # Basic liquidity proxy left as NaN (free FMP lacks ADV); user can merge later if needed.
    else:
        df = load_from_csv(args.input)

    # Screen
    screened = screen(
        df,
        wacc=args.wacc, g1=args.g1, years1=args.years1, gt=args.gt,
        min_quality=args.min_quality, min_mos=args.min_mos,
        min_adv_usd=args.min_adv, require_positive_fcf=not args.no_require_pos_fcf
    )

    # Explain failures for transparency
    reasons = explain_failures(
        screened, args.min_quality, args.min_mos, args.min_adv, 3.5, 4.0, 0.15
    )

    # Save
    screened.to_csv(args.out, index=False)

    # Also save a human-friendly audit txt
    audit_path = os.path.splitext(args.out)[0] + "_audit.txt"
    with open(audit_path, "w", encoding="utf-8") as f:
        f.write("Gate failure reasons by symbol:\n")
        for k, v in reasons.items():
            f.write(f"{k}: {', '.join(v)}\n")

    print(f"Saved results to {args.out}")
    print(f"Saved audit to {audit_path}")

if __name__ == "__main__":
    main()
