import argparse
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any

import requests

BASE_URL = "https://fapi.binance.com"

# ====== 如果你需要代理，这里填上；否则设为 None ======
PROXIES = None
# PROXIES = {
#     "http": "http://127.0.0.1:7890",
#     "https": "http://127.0.0.1:7890",
# }


# ========== 工具函数 ==========

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
    把 "1m","5m","1h","4h","1d" 这类转成毫秒
    """
    unit = interval[-1]
    num = int(interval[:-1])
    if unit == "m":  # 分钟
        return num * 60 * 1000
    if unit == "h":  # 小时
        return num * 60 * 60 * 1000
    if unit == "d":  # 天
        return num * 24 * 60 * 60 * 1000
    raise ValueError(f"Unsupported interval: {interval}")


def fetch_all_klines(
    session: requests.Session,
    path: str,
    params_base: Dict[str, Any],
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
) -> List[List[Any]]:
    """
    通用 K 线拉取器，自动用 startTime 分页，直到 end_ms 或数据拉完。
    适用于:
      - /fapi/v1/indexPriceKlines
      - /fapi/v1/markPriceKlines
    """
    all_rows: List[List[Any]] = []
    cur_start = start_ms

    while cur_start < end_ms:
        params = dict(params_base)
        params.update({
            "startTime": cur_start,
            "endTime": end_ms,
            "limit": limit,
        })
        data = http_get(session, path, params)
        if not data:
            break

        all_rows.extend(data)

        last_open = data[-1][0]  # open time
        if last_open >= end_ms or len(data) < limit:
            break

        # 下一个窗口从 last_open 之后开始，避免重复
        cur_start = last_open + 1

        # 稍微 sleep，避免太快打满限速（你也可以关掉）
        time.sleep(0.05)

    return all_rows


# ========== 核心逻辑 ==========

def load_futures_symbols_for_pair(session: requests.Session, pair: str,
                                  target_contract_types: List[str]):
    """
    从 /fapi/v1/exchangeInfo 中筛选出:
    - 对应 pair（如 BTCUSDT）
    - 指定的 contractType（CURRENT_QUARTER / NEXT_QUARTER）
    返回每个合约: symbol, contractType, onboardDate, deliveryDate
    """
    data = http_get(session, "/fapi/v1/exchangeInfo")
    if not data:
        raise RuntimeError("无法获取 exchangeInfo")

    res = []
    for sym in data.get("symbols", []):
        if sym.get("pair") != pair:
            continue
        ctype = sym.get("contractType")
        if ctype not in target_contract_types:
            continue
        # futures 的 Symbol 里会有 deliveryDate / onboardDate 字段
        delivery = int(sym.get("deliveryDate", 0))
        onboard = int(sym.get("onboardDate", 0))
        if delivery <= 0 or onboard <= 0:
            continue

        res.append({
            "symbol": sym["symbol"],
            "contractType": ctype,
            "deliveryDate": delivery,
            "onboardDate": onboard,
            "status": sym.get("status", ""),
        })
    return res


def backtest_term_basis(
    pair: str = "BTCUSDT",
    contract_types: List[str] = None,
    days: int = 365,
    interval: str = "4h",
    ann_threshold: float = 0.20,
):
    """
    回测逻辑：
    - 对过去 N 天内仍在存续/交割的所有季度合约，逐根 K 线计算年化基差
    - 统计年化 >= ann_threshold 的机会次数
    """
    if contract_types is None:
        contract_types = ["CURRENT_QUARTER"]

    session = requests.Session()

    now = datetime.now(timezone.utc)
    end_ms = int(now.timestamp() * 1000)
    start_ms = int((now - timedelta(days=days)).timestamp() * 1000)

    print(f"[*] 回测标的: pair={pair}, 合约类型={contract_types}, "
          f"区间=过去 {days} 天, 周期={interval}, 阈值={ann_threshold:.2f} (即 {ann_threshold*100:.1f}%)")

    # 1) 读取合约基本信息（deliveryDate / onboardDate）
    symbols_meta = load_futures_symbols_for_pair(session, pair, contract_types)
    if not symbols_meta:
        print("[!] 未找到任何匹配的季度合约，请检查 pair 或 contractTypes")
        return

    print("[*] 发现以下季度合约（含历史）：")
    for s in symbols_meta:
        d_dt = datetime.fromtimestamp(s["deliveryDate"] / 1000, tz=timezone.utc)
        o_dt = datetime.fromtimestamp(s["onboardDate"] / 1000, tz=timezone.utc)
        print(f"  - symbol={s['symbol']:<15} type={s['contractType']:<16} "
              f"onboard={o_dt.date()}  delivery={d_dt.date()}  status={s['status']}")

    # 2) 扣掉与回测区间完全不相交的合约
    active_symbols = []
    for s in symbols_meta:
        if s["deliveryDate"] < start_ms:
            continue  # 交割时间在回测开始前 -> 不需要
        if s["onboardDate"] > end_ms:
            continue  # 上线时间在回测结束后 -> 不需要
        active_symbols.append(s)

    if not active_symbols:
        print("[!] 在过去一年内没有任何相关季度合约（可能是合约刚上线不久？）")
        return

    print("\n[*] 参与本次回测的合约：")
    for s in active_symbols:
        d_dt = datetime.fromtimestamp(s["deliveryDate"] / 1000, tz=timezone.utc)
        o_dt = datetime.fromtimestamp(s["onboardDate"] / 1000, tz=timezone.utc)
        print(f"  - {s['symbol']} ({s['contractType']}), {o_dt.date()} ~ {d_dt.date()}")

    # 3) 拉 indexPriceKlines（指数价格，用作“现货指数 S”）
    print("\n[*] 拉取指数价 K 线 (indexPriceKlines)...")
    idx_params = {
        "pair": pair,
        "interval": interval,
    }
    index_rows = fetch_all_klines(
        session, "/fapi/v1/indexPriceKlines", idx_params, start_ms, end_ms
    )
    if not index_rows:
        print("[!] 未获取到指数 K 线数据，终止")
        return

    # 建一个 openTime -> closePrice 的 map
    index_map: Dict[int, float] = {}
    for row in index_rows:
        # [ openTime, open, high, low, close, ... ]
        t = int(row[0])
        close_price = float(row[4])
        index_map[t] = close_price

    print(f"[*] 指数 K 线条数: {len(index_map)}")

    # 4) 对每个季度合约，拉 markPriceKlines，然后计算年化基差
    interval_ms = interval_to_ms(interval)
    records = []  # 存所有时间点的计算结果

    for s in active_symbols:
        symbol = s["symbol"]
        ctype = s["contractType"]
        onboard = max(s["onboardDate"], start_ms)
        delivery = min(s["deliveryDate"], end_ms)

        print(f"\n[*] 处理合约 {symbol} ({ctype}), 有效区间: "
              f"{datetime.fromtimestamp(onboard/1000, tz=timezone.utc)} ~ "
              f"{datetime.fromtimestamp(delivery/1000, tz=timezone.utc)}")

        if delivery <= onboard:
            print("    区间为空，跳过")
            continue

        mk_params = {
            "symbol": symbol,
            "interval": interval,
        }
        mark_rows = fetch_all_klines(
            session, "/fapi/v1/markPriceKlines", mk_params, onboard, delivery
        )
        print(f"    拉到 markPriceKlines 条数: {len(mark_rows)}")

        for row in mark_rows:
            # K 线结构：[ openTime, open, high, low, close, ... ]
            t = int(row[0])
            f_close = float(row[4])
            s_close = index_map.get(t)
            if s_close is None or s_close <= 0:
                continue

            basis = f_close - s_close
            basis_rate = basis / s_close

            # 距离交割的天数
            t_days = (delivery - t) / 86_400_000.0
            if t_days <= 0:
                continue

            ann = basis_rate * (365.0 / t_days)

            records.append({
                "open_time": t,
                "symbol": symbol,
                "contractType": ctype,
                "S": s_close,
                "F": f_close,
                "basis": basis,
                "basis_rate": basis_rate,
                "ann": ann,
                "delivery": delivery,
            })

    if not records:
        print("\n[!] 没有任何可用数据点（可能是指数和期货 K 线未能对齐），终止")
        return

    # 5) 统计年化 > X% 的机会次数（按 K 线粒度）
    records.sort(key=lambda x: x["open_time"])
    total_points = len(records)
    pos_points = [r for r in records if r["ann"] >= ann_threshold]
    neg_points = [r for r in records if r["ann"] <= -ann_threshold]

    print("\n========== 回测结果（按单根 K 线统计） ==========")
    print(f"总有效数据点数: {total_points}")
    print(f"年化 >= {ann_threshold:.2f} ({ann_threshold*100:.1f}%) 的 K 线数量: {len(pos_points)}")
    print(f"年化 <= -{ann_threshold:.2f} (-{ann_threshold*100:.1f}%) 的 K 线数量: {len(neg_points)}")

    if pos_points:
        ann_vals = [r["ann"] for r in pos_points]
        print(f"正向机会年化分布: min={min(ann_vals):.2f}, "
              f"median={sorted(ann_vals)[len(ann_vals)//2]:.2f}, max={max(ann_vals):.2f}")

    # 6) 进一步：把“连续超阈值的 K 线”合并为一段机会
    def count_episodes(points: List[Dict[str, Any]], label: str):
        if not points:
            print(f"\n{label}: 没有满足阈值的 K 线，无法构造机会区间")
            return

        episodes = 0
        last_t = None
        for r in points:
            t = r["open_time"]
            if (last_t is None) or (t - last_t > 1.5 * interval_ms):
                # 新的一段机会
                episodes += 1
            last_t = t

        print(f"\n{label}:")
        print(f"  超阈值 K 线总数: {len(points)}")
        print(f"  估算独立机会区间数: {episodes}")

        # 顺便打印前几段的时间范围看看
        ep_ranges = []
        cur_start = None
        cur_end = None
        last_t = None
        for r in points:
            t = r["open_time"]
            if (last_t is None) or (t - last_t > 1.5 * interval_ms):
                if cur_start is not None:
                    ep_ranges.append((cur_start, cur_end))
                cur_start = t
            cur_end = t
            last_t = t
        if cur_start is not None:
            ep_ranges.append((cur_start, cur_end))

        print("  示例机会区间（最多展示前 5 段）:")
        for i, (st, ed) in enumerate(ep_ranges[:5], start=1):
            st_dt = datetime.fromtimestamp(st / 1000, tz=timezone.utc)
            ed_dt = datetime.fromtimestamp(ed / 1000, tz=timezone.utc)
            print(f"    {i}. {st_dt}  ~  {ed_dt}")

    count_episodes(pos_points, f"正向机会 (ann >= {ann_threshold:.2f})")
    count_episodes(neg_points, f"反向机会 (ann <= -{ann_threshold:.2f})")

    print("\n[*] 回测完成。你可以通过修改 ann_threshold / interval / days 来重新评估阈值可行性。")


# ========== 命令行入口 ==========

def main():
    parser = argparse.ArgumentParser(
        description="Binance 期限套利机会回测（基于季度合约年化基差）"
    )
    parser.add_argument("--pair", default="BTCUSDT", help="例: BTCUSDT / ETHUSDT")
    parser.add_argument("--days", type=int, default=365, help="回测天数，默认 365")
    parser.add_argument("--interval", default="1h",
                        help="K 线周期，如 1h / 4h / 1d")
    parser.add_argument("--threshold", type=float, required=True, default="0.2",
                        help="年化阈值（小数），例如 0.2 表示 20%%")
    parser.add_argument("--contract-types", nargs="+",
                        default=["CURRENT_QUARTER"],
                        help="合约类型列表，默认只看 CURRENT_QUARTER；"
                             "可加 NEXT_QUARTER，如: --contract-types CURRENT_QUARTER NEXT_QUARTER")

    args = parser.parse_args()

    backtest_term_basis(
        pair=args.pair,
        contract_types=args.contract_types,
        days=args.days,
        interval=args.interval,
        ann_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()

"""
使用方式：（CURRENT_QUARTER和NEXT_QUARTER）
python3 binance_term_arb_backtest.py \
  --threshold 0.2 \
  --contract-types CURRENT_QUARTER NEXT_QUARTER
"""