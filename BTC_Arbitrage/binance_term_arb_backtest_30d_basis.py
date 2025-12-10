import argparse
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List

import requests

BASE_URL = "https://fapi.binance.com"

# 如果你需要代理，在这里填上；没有就保持 None
PROXIES = None
# PROXIES = {
#     "http": "http://127.0.0.1:7890",
#     "https": "http://127.0.0.1:7890",
# }


# ========= 通用 HTTP 工具 =========

def http_get(session: requests.Session, path: str, params: Dict[str, Any] = None):
    url = BASE_URL + path
    try:
        resp = session.get(url, params=params or {}, timeout=10, proxies=PROXIES)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[ERROR] GET {path} params={params} failed: {e}")
        return None


def interval_to_ms(interval: str) -> int:
    """
    "5m","15m","1h","4h","1d" -> 毫秒
    """
    unit = interval[-1]
    num = int(interval[:-1])
    if unit == "m":
        return num * 60 * 1000
    if unit == "h":
        return num * 60 * 60 * 1000
    if unit == "d":
        return num * 24 * 60 * 60 * 1000
    raise ValueError(f"Unsupported interval: {interval}")


# ========= 业务相关 =========

def load_delivery_map(session: requests.Session, pair: str, contract_types: List[str]):
    """
    从 /fapi/v1/exchangeInfo 拿每个 contractType 的 deliveryDate
    """
    data = http_get(session, "/fapi/v1/exchangeInfo")
    if not data:
        raise RuntimeError("无法获取 exchangeInfo")

    delivery_map: Dict[str, int] = {}
    symbol_map: Dict[str, str] = {}

    for sym in data.get("symbols", []):
        if sym.get("pair") != pair:
            continue
        ctype = sym.get("contractType")
        if ctype not in contract_types:
            continue
        delivery = int(sym.get("deliveryDate", 0))
        if delivery <= 0:
            continue
        delivery_map[ctype] = delivery
        symbol_map[ctype] = sym.get("symbol")

    return delivery_map, symbol_map


def fetch_all_basis(
    session: requests.Session,
    pair: str,
    contract_type: str,
    period: str,
    start_ms: int,
    end_ms: int,
    limit: int = 500,
):
    """
    用 /futures/data/basis 把近 30 天内的数据分页拉完。
    文档：GET /futures/data/basis :contentReference[oaicite:1]{index=1}
    """
    all_rows = []
    cur_start = start_ms
    last_ts = None

    while cur_start < end_ms:
        params = {
            "pair": pair,
            "contractType": contract_type,
            "period": period,
            "limit": limit,
            "startTime": cur_start,
            "endTime": end_ms,
        }
        data = http_get(session, "/futures/data/basis", params)
        if not data:
            break

        all_rows.extend(data)

        last = data[-1]
        ts = int(last["timestamp"])
        if last_ts is not None and ts <= last_ts:
            break  # 防无限循环
        last_ts = ts

        if ts >= end_ms or len(data) < limit:
            break

        cur_start = ts + 1
        time.sleep(0.05)

    return all_rows


def compute_ann_from_row(row: Dict[str, Any], delivery_ms: int):
    """
    优先用 annualizedBasisRate，如果是空字符串，则按 basisRate + 剩余天数自己算。
    """
    ann_raw = row.get("annualizedBasisRate", "")
    if isinstance(ann_raw, str) and ann_raw != "":
        try:
            return float(ann_raw)
        except ValueError:
            pass

    # 自己算
    ts = int(row["timestamp"])
    basis_rate = float(row["basisRate"])
    if delivery_ms <= ts:
        return None
    t_days = (delivery_ms - ts) / 86_400_000.0
    if t_days <= 0:
        return None
    return basis_rate * (365.0 / t_days)


def backtest_last_month(
    pair: str = "BTCUSDT",
    contract_types: List[str] = None,
    period: str = "1h",          # 你可以改成 "5m" 更高频
    days: int = 30,              # 最大只能到 30
    ann_threshold: float = 0.20, # 例如 0.2 = 20%
):
    if contract_types is None:
        contract_types = ["CURRENT_QUARTER"]

    days = min(days, 30)  # 接口本身只支持最近 30 天 :contentReference[oaicite:2]{index=2}

    session = requests.Session()
    now = datetime.now(timezone.utc)
    end_ms = int(now.timestamp() * 1000)
    start_ms = int((now - timedelta(days=days)).timestamp() * 1000)

    print(f"[*] 回测标的: pair={pair}, 合约类型={contract_types}, "
          f"区间=最近 {days} 天, 周期={period}, 阈值={ann_threshold:.2f} ({ann_threshold*100:.1f}%)")

    # 1) 拿交割时间
    delivery_map, symbol_map = load_delivery_map(session, pair, contract_types)
    print("[*] 交割信息：")
    for c in contract_types:
        symbol = symbol_map.get(c, "N/A")
        d_ms = delivery_map.get(c)
        if d_ms:
            d_dt = datetime.fromtimestamp(d_ms / 1000, tz=timezone.utc)
            print(f"  - {c:<16} symbol={symbol:<15} delivery={d_dt}")
        else:
            print(f"  - {c:<16} symbol={symbol:<15} delivery=未知")

    # 2) 拉 basis 历史
    interval_ms = interval_to_ms(period)
    records = []

    for c in contract_types:
        if c not in delivery_map:
            print(f"[!] {c} 无交割时间，跳过")
            continue

        print(f"\n[*] 拉取 {c} 的 basis 历史 ...")
        rows = fetch_all_basis(session, pair, c, period, start_ms, end_ms)
        print(f"    共获取 {len(rows)} 条 basis 记录")

        for row in rows:
            ts = int(row["timestamp"])
            idx = float(row["indexPrice"])
            fut = float(row["futuresPrice"])
            basis = float(row["basis"])
            basis_rate = float(row["basisRate"])
            ann = compute_ann_from_row(row, delivery_map[c])
            if ann is None:
                continue

            records.append({
                "timestamp": ts,
                "contractType": c,
                "pair": pair,
                "indexPrice": idx,
                "futuresPrice": fut,
                "basis": basis,
                "basis_rate": basis_rate,
                "ann": ann,
            })

    if not records:
        print("\n[!] 近 30 天没有任何可用的 basis 数据")
        return

    records.sort(key=lambda x: x["timestamp"])
    total_points = len(records)
    pos_points = [r for r in records if r["ann"] >= ann_threshold]
    neg_points = [r for r in records if r["ann"] <= -ann_threshold]

    print("\n========== 回测结果（近一月，按单个时间点统计） ==========")
    print(f"总数据点数: {total_points}")
    print(f"年化 >= {ann_threshold:.2f} ({ann_threshold*100:.1f}%) 的点数: {len(pos_points)}")
    print(f"年化 <= -{ann_threshold:.2f} (-{ann_threshold*100:.1f}%) 的点数: {len(neg_points)}")

    if pos_points:
        ann_vals = [r["ann"] for r in pos_points]
        ann_vals_sorted = sorted(ann_vals)
        mid = ann_vals_sorted[len(ann_vals_sorted)//2]
        print(f"正向机会年化分布: min={min(ann_vals):.2f}, median={mid:.2f}, max={max(ann_vals):.2f}")

    # 把连续点合并成“机会区间”
    def count_episodes(points: List[Dict[str, Any]], label: str):
        if not points:
            print(f"\n{label}: 无")
            return

        points_sorted = sorted(points, key=lambda x: x["timestamp"])
        episodes = 0
        last_t = None
        ep_ranges = []
        cur_start = None
        cur_end = None

        for r in points_sorted:
            t = r["timestamp"]
            if (last_t is None) or (t - last_t > 1.5 * interval_ms):
                # 新的一段
                if cur_start is not None:
                    ep_ranges.append((cur_start, cur_end))
                cur_start = t
            cur_end = t
            last_t = t

        if cur_start is not None:
            ep_ranges.append((cur_start, cur_end))

        episodes = len(ep_ranges)
        print(f"\n{label}:")
        print(f"  超阈值数据点: {len(points_sorted)}")
        print(f"  估算独立机会区间数: {episodes}")
        print("  示例区间（最多展示前 5 段）:")
        for i, (st, ed) in enumerate(ep_ranges[:5], start=1):
            st_dt = datetime.fromtimestamp(st / 1000, tz=timezone.utc)
            ed_dt = datetime.fromtimestamp(ed / 1000, tz=timezone.utc)
            print(f"    {i}. {st_dt}  ~  {ed_dt}")

    count_episodes(pos_points, f"正向期限套利机会 (ann >= {ann_threshold:.2f})")
    count_episodes(neg_points, f"反向机会 (ann <= -{ann_threshold:.2f})")

    print("\n[*] 回测完成，你可以调整 threshold / period 再多跑几次。")


# ========= 命令行入口 =========

def main():
    parser = argparse.ArgumentParser(
        description="Binance 最近一月期限套利机会回测（基于 /futures/data/basis）"
    )
    parser.add_argument("--pair", default="BTCUSDT", help="交易对，例如 BTCUSDT")
    parser.add_argument("--period", default="1h",
                        help="basis 时间粒度: 5m / 15m / 30m / 1h / 2h / 4h / 6h / 12h / 1d")
    parser.add_argument("--days", type=int, default=30,
                        help="回测天数，最大 30")
    parser.add_argument("--threshold", type=float, required=True,
                        help="年化阈值（小数），例如 0.2 表示 20%%")
    parser.add_argument("--contract-types", nargs="+",
                        default=["CURRENT_QUARTER"],
                        help="合约类型，例如: CURRENT_QUARTER NEXT_QUARTER PERPETUAL（一般期限套利只看季度）")

    args = parser.parse_args()

    backtest_last_month(
        pair=args.pair,
        contract_types=args.contract_types,
        period=args.period,
        days=args.days,
        ann_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()

"""
使用方式：

python3 binance_term_arb_backtest_30d_basis.py \
  --threshold 0.07 \
  --period 5m \
  --contract-types CURRENT_QUARTER NEXT_QUARTER

"""