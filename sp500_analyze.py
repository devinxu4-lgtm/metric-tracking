#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# 基于 SQLite 分析（推荐）
python3 sp500_analyze.py --db metrics.db --require-value --codes-only --top 50

# 想导出到 CSV（带深折价标记）
python3 sp500_analyze.py --db metrics.db --output picks.csv

# 如果你先从 DB 导出成 CSV 再分析（完全离线不依赖 sqlite3）
python3 sp500_analyze.py --input metrics.csv --codes-only

调整阈值
python3 sp500_analyze.py --db metrics.db \
  --min-roic 0.15 --value-ev-ebit-max 10 --value-fcf-yield-min 0.06 \
  --require-value --codes-only
"""

"""
S&P500 价值风格选股 · 离线分析部分 (sp500_analyze.py)

- 从 SQLite (metrics.db) 或 CSV/JSON 文件读取已抓取好的指标数据
- 应用“财务质量很好”的一组阈值 + 可选“深折价”规则
- 输出极简：仅代码 (--codes-only) 或 `symbol,deep` 两列，可限制 top N，可写入 CSV

配合抓取脚本使用：
    python3 sp500_fetch.py  --proxy http://127.0.0.1:7890 --db metrics.db ...
    python3 sp500_analyze.py --db metrics.db --require-value --codes-only --top 50
"""

import os
import sys
import csv
import json
import math
import argparse
import sqlite3
from typing import List, Dict, Any, Optional

def _eprint(msg: str) -> None:
    try:
        sys.stderr.write(str(msg) + "\n")
    except Exception:
        pass

# ---- 默认阈值（可通过命令行覆盖） ---- #
DEFAULTS: Dict[str, float] = {
    "min_roic": 0.12,              # ROIC ≥ 12%
    "min_gross_margin": 0.30,      # 毛利率 ≥ 30%
    "min_net_margin": 0.10,        # 净利率 ≥ 10%
    "max_net_debt_to_ebitda": 2.5, # 净负债 / EBITDA ≤ 2.5
    "min_interest_coverage": 6.0,  # 利息保障倍数 ≥ 6
    "min_fcf_yield": 0.02,         # FCF Yield ≥ 2% (基本质量)
    # 深折价规则：三选二
    "value_ev_ebit_max": 12.0,     # EV/EBIT ≤ 12
    "value_pe_max": 20.0,          # P/E ≤ 20
    "value_fcf_yield_min": 0.05,   # FCF Yield ≥ 5%
}

# ---- 质量与深折价规则 ---- #
class Rule:
    def __init__(self, name: str, fn):
        self.name = name
        self.fn = fn
    def __call__(self, m: Dict[str, Any]) -> bool:
        try:
            return bool(self.fn(m))
        except Exception:
            return False

def quality_rules(cfg: Dict[str, float]) -> List[Rule]:
    return [
        Rule("ROIC",            lambda m: m.get("roic") is not None and m["roic"] >= cfg["min_roic"]),
        Rule("GrossMargin",     lambda m: m.get("gross_margin") is not None and m["gross_margin"] >= cfg["min_gross_margin"]),
        Rule("NetMargin",       lambda m: m.get("net_margin") is not None and m["net_margin"] >= cfg["min_net_margin"]),
        Rule("NetDebt/EBITDA",  lambda m: m.get("net_debt_to_ebitda") is not None and m["net_debt_to_ebitda"] <= cfg["max_net_debt_to_ebitda"]),
        Rule("InterestCoverage",lambda m: m.get("interest_coverage") is not None and m["interest_coverage"] >= cfg["min_interest_coverage"]),
        Rule("FCFYield",        lambda m: m.get("fcf_yield") is not None and m["fcf_yield"] >= cfg["min_fcf_yield"]),
    ]

def deep_value_flag(m: Dict[str, Any], cfg: Dict[str, float]) -> int:
    """
    深折价标记：三选二
      - EV/EBIT 足够便宜
      - FCF Yield 足够高
      - P/E 足够低
    任意两条满足则视为 deep=1，否则 0
    """
    ok_ev  = (m.get("ev_ebit")   is not None) and (m["ev_ebit"]   <= cfg["value_ev_ebit_max"])
    ok_fcf = (m.get("fcf_yield") is not None) and (m["fcf_yield"] >= cfg["value_fcf_yield_min"])
    ok_pe  = (m.get("pe")        is not None) and (m["pe"]        <= cfg["value_pe_max"])

    cnt = int(ok_ev) + int(ok_fcf) + int(ok_pe)
    return 1 if cnt >= 2 else 0

# ---- 数据加载：SQLite 或 CSV/JSON ---- #
def load_from_db(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        _eprint(f"[ERROR] 数据库不存在：{path}")
        return []
    conn = sqlite3.connect(path)
    cols = [
        "symbol",
        "roic",
        "gross_margin",
        "net_margin",
        "net_debt_to_ebitda",
        "interest_coverage",
        "fcf_yield",
        "ev_ebit",
        "pe",
    ]
    cur = conn.execute("SELECT " + ",".join(cols) + " FROM metrics")
    rows = [{k: v for k, v in zip(cols, r)} for r in cur.fetchall()]
    conn.close()
    return rows

def load_from_file(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        _eprint(f"[ERROR] 文件不存在：{path}")
        return []
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv"):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [dict(r) for r in reader]
    if ext in (".json", ".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("rows", [])
    _eprint(f"[ERROR] 不支持的文件类型：{path}")
    return []

# ---- 主流程 ---- #
def analyze_main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="S&P500 价值风格离线分析")
    ap.add_argument("--db", type=str, default="", help="SQLite 路径（优先）")
    ap.add_argument("--input", type=str, default="", help="CSV/JSON 文件路径")
    ap.add_argument("--top", type=int, default=0, help="只取前 N 个（按代码排序后）")
    ap.add_argument("--require-value", action="store_true", help="只保留深折价股票")
    ap.add_argument("--codes-only", action="store_true", help="仅输出股票代码")
    ap.add_argument("--output", type=str, default="", help="写入 CSV，而不是打印到 stdout")

    # 阈值可调
    ap.add_argument("--min-roic", type=float, default=DEFAULTS["min_roic"])
    ap.add_argument("--min-gross-margin", type=float, default=DEFAULTS["min_gross_margin"])
    ap.add_argument("--min-net-margin", type=float, default=DEFAULTS["min_net_margin"])
    ap.add_argument("--max-net-debt-to-ebitda", type=float, default=DEFAULTS["max_net_debt_to_ebitda"])
    ap.add_argument("--min-interest-coverage", type=float, default=DEFAULTS["min_interest_coverage"])
    ap.add_argument("--min-fcf-yield", type=float, default=DEFAULTS["min_fcf_yield"])
    ap.add_argument("--value-ev-ebit-max", type=float, default=DEFAULTS["value_ev_ebit_max"])
    ap.add_argument("--value-pe-max", type=float, default=DEFAULTS["value_pe_max"])
    ap.add_argument("--value-fcf-yield-min", type=float, default=DEFAULTS["value_fcf_yield_min"])

    args = ap.parse_args(argv)

    cfg = DEFAULTS.copy()
    # 把命令行参数覆盖到 cfg
    for k, v in vars(args).items():
        key = k.replace("-", "_")
        if key in cfg and v is not None:
            cfg[key] = v

    rows: List[Dict[str, Any]] = []
    if args.db:
        rows = load_from_db(args.db)
    elif args.input:
        rows = load_from_file(args.input)
    else:
        _eprint("[ERROR] 请选择 --db 或 --input")
        return 2

    def _to_float(x: Any) -> Optional[float]:
        try:
            if x is None:
                return None
            if isinstance(x, str) and x.strip() == "":
                return None
            v = float(x)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        except Exception:
            return None

    # 归一化字段
    norm: List[Dict[str, Any]] = []
    for r in rows:
        m: Dict[str, Any] = {k: None for k in [
            "symbol",
            "roic",
            "gross_margin",
            "net_margin",
            "net_debt_to_ebitda",
            "interest_coverage",
            "fcf_yield",
            "ev_ebit",
            "pe",
        ]}
        m["symbol"] = str(r.get("symbol", "")).upper()
        for k in [
            "roic",
            "gross_margin",
            "net_margin",
            "net_debt_to_ebitda",
            "interest_coverage",
            "fcf_yield",
            "ev_ebit",
            "pe",
        ]:
            m[k] = _to_float(r.get(k))
        if m["symbol"]:
            norm.append(m)

    if not norm:
        _eprint("[ERROR] 没有有效记录")
        return 2

    # 质量过滤
    rules = quality_rules(cfg)
    filtered = [m for m in norm if all(rule(m) for rule in rules)]
    if args.require_value:
        filtered = [m for m in filtered if deep_value_flag(m, cfg) == 1]

    # 按代码排序，方便稳定 diff
    filtered = sorted(filtered, key=lambda x: x.get("symbol", ""))

    if args.top > 0:
        filtered = filtered[: args.top]

    # 输出
    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if args.codes_only:
                w.writerow(["symbol"])
                for m in filtered:
                    w.writerow([m["symbol"]])
            else:
                w.writerow(["symbol", "deep"])
                for m in filtered:
                    w.writerow([m["symbol"], deep_value_flag(m, cfg)])
    else:
        if args.codes_only:
            for m in filtered:
                print(m["symbol"])
        else:
            for m in filtered:
                print(f"{m['symbol']},{deep_value_flag(m, cfg)}")

    return 0

if __name__ == "__main__":
    sys.exit(analyze_main())
