#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令方式：
# 先把旧库备份一下（可选）
cp metrics.db metrics_old.db

# 全量重抓（不用 --resume）
python3 sp500_fetch.py --proxy http://127.0.0.1:7890 \
  --db metrics.db \
  --max-workers 1 --qps 0.33 --retries 6
"""

"""
S&P500 抓取并持久化到 SQLite（“抓取部分”）
- 代理支持 (--proxy)，维基/成分抓取与 yfinance 共用环境代理
- 强制禁用 yfinance 的 curl_cffi 后端，避免本机 libcurl/OpenSSL 不兼容导致 TLS 错误 (curl:35)
- 轻量并发、进度条；失败默认跳过并继续；支持 --resume 只补旧
- **新增** 防限流：全局 QPS 限速 + 429/Too Many Requests 指数退避重试 + 可打散分片抓取
- 输出持久化到 SQLite (metrics.db)，表 metrics(symbol 主键, 指标列, updated_at)
"""

import os, sys, re, time, threading, argparse, sqlite3, random
from typing import List, Dict, Any, Optional

# ---- 基础设置：强制禁用 curl_cffi，避免本机 libcurl/OpenSSL 不兼容 ---- #
os.environ['YF_USE_CURL_CFFI'] = '0'

def _eprint(msg: str) -> None:
    try:
        sys.stderr.write(str(msg) + "\n")
    except Exception:
        pass

# ---- 代理注入（给 yfinance/urllib 共用） ---- #
def configure_proxy(proxy_url: str, install_for_urllib: bool = True) -> None:
    if not proxy_url:
        return
    for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:
        os.environ[k] = proxy_url
    if install_for_urllib:
        try:
            import urllib.request as _ur
            opener = _ur.build_opener(_ur.ProxyHandler({'http': proxy_url, 'https': proxy_url}))
            _ur.install_opener(opener)
        except Exception as e:
            _eprint(f"[WARN] 安装 urllib 代理失败：{e}")

# ---- 进度条（标准库） ---- #
class Progress:
    def __init__(self, total:int, label:str='fetch', enabled:bool=True, file=None, min_interval:float=0.1):
        self.t = max(0,total); self.l=label; self.e=enabled; self.f=file or sys.stderr
        self.mi=min_interval; self.d=0; self.ok=0; self.fail=0; self.st=time.monotonic(); self.la=0.0
        self.lock=threading.Lock()
        if self.e:
            try: self.f.write(f"[{self.l}] 0/{self.t} (0.0%)\r"); self.f.flush()
            except Exception: self.e=False
    def upd(self, ok:Optional[bool]=None):
        with self.lock:
            self.d+=1; self.ok+= (ok is True); self.fail+= (ok is False)
            if not self.e: return
            now=time.monotonic()
            if (now-self.la)<self.mi and self.d<self.t: return
            pct=(self.d/self.t*100.0) if self.t else 100.0
            eta=((now-self.st)/max(1,self.d)*(self.t-self.d)) if self.d else 0.0
            msg=f"[{self.l}] {self.d}/{self.t} ({pct:0.1f}%) OK:{self.ok} FAIL:{self.fail} ETA:{eta:0.1f}s"
            try:
                self.f.write("\r"+msg);  
                if self.d>=self.t: self.f.write("\n")
                self.f.flush()
            except Exception: self.e=False
            self.la=now
    def close(self):
        if self.e:
            try: self.f.write("\n"); self.f.flush()
            except Exception: pass

# ---- yfinance / 成分获取 ---- #
try:
    import yfinance as yf
    import pandas as pd  # 仅用于空表判断与自测
    HAS_YF=True
except Exception:
    HAS_YF=False
    pd = None  # type: ignore

_SP500_WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def _parse_sp500_from_html(html:str)->List[str]:
    m = re.search(r"(?is)<table[^>]*>(?:(?!</table>).)*?<th[^>]*>\s*Symbol\s*</th>(?:(?!</table>).)*?</table>", html)
    table = m.group(0) if m else html
    out:List[str]=[]
    for row in re.finditer(r"(?is)<tr[^>]*>(.*?)</tr>", table):
        c = row.group(1)
        m2 = re.search(r"(?is)<td[^>]*>(.*?)</td>", c)
        if not m2:
            continue
        raw = re.sub(r"<[^>]+>", "", m2.group(1)).strip().upper()
        if not raw:
            continue
        s = re.sub(r"[^A-Z0-9\-]", "", raw.replace(".", "-"))
        if not s:
            continue
        # 过滤明显不是股票代码的噪声（例如 MAY12014 这种由 “May 1, 2014” 转来的日期串）
        # 当前 S&P500 正规代码长度 ≤ 5（BRK-B、GOOGL 等），故 >5 直接丢弃
        if len(s) > 5:
            continue
        # 排除纯数字串
        if s.isdigit():
            continue
        out.append(s)
    seen=set(); res=[]
    for s in out:
        if s not in seen: seen.add(s); res.append(s)
    return res

def _get_sp500_from_wikipedia(timeout:float=15.0)->List[str]:
    try:
        import urllib.request as ur
        req = ur.Request(_SP500_WIKI, headers={"User-Agent":"Mozilla/5.0"})
        with ur.urlopen(req, timeout=timeout) as r:
            html = r.read().decode("utf-8", errors="ignore")
        syms = _parse_sp500_from_html(html)
        return sorted(set(syms))
    except Exception as e:
        _eprint(f"[WARN] Wikipedia 获取失败：{e}"); return []

def get_sp500_symbols()->List[str]:
    if HAS_YF and hasattr(yf, 'tickers_sp500'):
        try:
            return sorted(set([str(s).upper().replace('.', '-') for s in yf.tickers_sp500()]))
        except Exception:
            pass
    syms=_get_sp500_from_wikipedia()
    if not syms:
        _eprint("[WARN] 无法获取 S&P500 成分，请改用 --symbols/--from-file 或稍后再试。")
    return syms

# ---- 安全选择非空 DataFrame（避免布尔求值歧义） ---- #
def _pick_df(primary, fallback):
    try:
        if primary is not None and getattr(primary, "empty", True) is False:
            return primary
    except Exception:
        pass
    return fallback

# ---- 指标计算 ---- #
def _sum_last_n(df, row:str, n:int=4)->Optional[float]:
    try:
        if df is None:
            return None
        s=df.loc[row].dropna()
        if s.empty: return None
        return float(s.iloc[:n].sum())
    except Exception:
        return None

def _sum_last_n_any(df, rows, n:int=4)->Optional[float]:
    """
    依次尝试多个行名，返回第一个非 None 的合计值。
    用于兼容 yfinance 不同标的之间稍微不一样的科目命名。
    """
    if df is None:
        return None
    for name in rows:
        v = _sum_last_n(df, name, n)
        if v is not None:
            return v
    return None

def fetch_metrics(sym:str)->Dict[str,Any]:
    t=yf.Ticker(sym)
    fast=getattr(t,'fast_info',{}) or {}
    mc= fast.get('market_cap') or fast.get('marketCap')

    fin=_pick_df(getattr(t,'quarterly_financials', None), getattr(t,'financials', None))
    cf =_pick_df(getattr(t,'quarterly_cashflow', None),    getattr(t,'cashflow', None))
    bs =_pick_df(getattr(t,'quarterly_balance_sheet', None),getattr(t,'balance_sheet', None))

    rev=_sum_last_n(fin,'Total Revenue'); gp=_sum_last_n(fin,'Gross Profit')
    ebit=_sum_last_n(fin,'Operating Income')
    ni=_sum_last_n(fin,'Net Income Common Stockholders') or _sum_last_n(fin,'Net Income')
    ie=_sum_last_n(fin,'Interest Expense')
    da=_sum_last_n(cf,'Depreciation') or 0.0
    ebitda=(ebit or 0.0)+(da or 0.0)
    cfo=_sum_last_n_any(cf, ['Total Cash From Operating Activities', 'Operating Cash Flow'])
    capex=_sum_last_n_any(cf, ['Capital Expenditures', 'Capital Expenditure'])
    fcf=float(cfo+capex) if (cfo is not None and capex is not None) else None
    td=None
    try:
        td=_sum_last_n(bs,'Total Debt',1)
        if td is None:
            td=( _sum_last_n(bs,'Short Long Term Debt',1) or 0.0 ) + ( _sum_last_n(bs,'Long Term Debt',1) or 0.0 )
    except Exception:
        td=None
    cash=None
    try:
        cash=_sum_last_n(bs,'Cash And Cash Equivalents',1) or _sum_last_n(bs,'Cash',1)
    except Exception:
        cash=None
    net_debt=(td or 0.0)-(cash or 0.0)
    pe = getattr(t,'info',{}).get('trailingPE') if hasattr(t,'info') else None
    if mc is None: mc = getattr(t,'info',{}).get('marketCap')
    ev=(mc or 0.0)+max(net_debt or 0.0,0.0)
    gm=(gp/rev) if (gp and rev) else None
    nm=(ni/rev) if (ni and rev) else None
    nde=(net_debt/ebitda) if (ebitda and ebitda>0) else None
    ic =(ebit/abs(ie)) if (ebit and ie and ie!=0) else None
    fcfy=(fcf/mc) if (fcf is not None and mc) else None
    ev_ebit=(ev/ebit) if (ebit and ebit>0) else None
    tax=0.21
    nopat=(ebit*(1-tax)) if ebit else None
    eq=None
    try:
        eq=_sum_last_n(bs,'Total Stockholder Equity',1) or _sum_last_n(bs,'Total Equity Gross Minority Interest',1)
    except Exception:
        eq=None
    icap=None
    if (td is not None) or (eq is not None):
        icap=(td or 0.0)+(eq or 0.0)-(cash or 0.0)
        if icap is not None and icap<=0: icap=None
    roic=(nopat/icap) if (nopat and icap) else None
    return {
        'symbol':sym,'roic':roic,'gross_margin':gm,'net_margin':nm,
        'net_debt_to_ebitda':nde,'interest_coverage':ic,
        'fcf_yield':fcfy,'ev_ebit':ev_ebit,'pe':pe
    }

# ---- SQLite 持久化 ---- #
_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS metrics("
    "symbol TEXT PRIMARY KEY,"
    "roic REAL, gross_margin REAL, net_margin REAL,"
    "net_debt_to_ebitda REAL, interest_coverage REAL,"
    "fcf_yield REAL, ev_ebit REAL, pe REAL,"
    "updated_at TEXT NOT NULL)"
)

def db_open(path:str)->sqlite3.Connection:
    conn=sqlite3.connect(path)
    conn.execute(_SCHEMA)
    return conn

def db_get_existing(conn)->Dict[str,float]:
    cur=conn.execute("SELECT symbol, strftime('%s',updated_at) FROM metrics")
    return {r[0]: float(r[1]) for r in cur.fetchall()}

def db_upsert(conn, row:Dict[str,Any])->None:
    cols=['symbol','roic','gross_margin','net_margin','net_debt_to_ebitda','interest_coverage','fcf_yield','ev_ebit','pe']
    placeholders=",".join([":" + c for c in cols])
    sql=("INSERT INTO metrics(" + ",".join(cols) + ",updated_at) VALUES(" + placeholders + ",datetime('now')) "
         "ON CONFLICT(symbol) DO UPDATE SET "
         + ",".join([f"{c}=excluded.{c}" for c in cols[1:]]) + ", updated_at=datetime('now')")
    conn.execute(sql, row)

# ---- 全局 QPS 限速器 ---- #
class QPSLimiter:
    def __init__(self, qps: float = 0.0, fixed_sleep: float = 0.0):
        self.qps = float(qps)
        self.fixed_sleep = float(fixed_sleep)
        self.min_interval = 1.0 / self.qps if self.qps > 0 else 0.0
        self.lock = threading.Lock()
        self.last = 0.0
    def wait(self):
        if self.fixed_sleep > 0:
            time.sleep(self.fixed_sleep)
        if self.min_interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            wait = max(0.0, self.min_interval - (now - self.last))
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()

# ---- 抓取主流程 ---- #
def fetch_main(argv:Optional[List[str]]=None)->int:
    ap=argparse.ArgumentParser(description='S&P500 抓取并持久化到 SQLite')
    ap.add_argument('--db', type=str, default='metrics.db', help='SQLite 文件路径')
    ap.add_argument('--symbols', type=str, default='', help='逗号分隔的自定义代码（优先）')
    ap.add_argument('--from-file', type=str, default='', help='从文本文件读取代码（每行一个，不含表头）')
    ap.add_argument('--resume', action='store_true', help='仅抓取缺失或过期的股票')
    ap.add_argument('--stale-days', type=int, default=7, help='过期天数阈值，配合 --resume 使用')
    ap.add_argument('--proxy', type=str, default='', help='HTTP(S) 代理，如 http://127.0.0.1:7890')
    ap.add_argument('--retries', type=int, default=5, help='每票最大重试次数（含首发）')
    ap.add_argument('--max-workers', type=int, default=2, help='并发度，建议 1~3 防限流')
    ap.add_argument('--dump-sp500', action='store_true', help='仅打印 S&P500 成分数量并退出')
    ap.add_argument('--no-progress', action='store_true')
    # 限流/退避/分片
    ap.add_argument('--qps', type=float, default=0.33, help='全局 QPS 限速（每秒请求数，0 关闭，默认 0.33≈每 3s 一票）')
    ap.add_argument('--sleep', type=float, default=0.0, help='每票固定额外 sleep 秒数')
    ap.add_argument('--shard', type=int, default=1, help='当前分片编号（从 1 开始）')
    ap.add_argument('--of', type=int, default=1, help='总分片数')
    ap.add_argument('--shuffle', action='store_true', help='随机打散抓取顺序（配合 --resume 减少热点）')
    ap.add_argument('--self-test', action='store_true', help='运行内置最小自测后退出')
    args=ap.parse_args(argv)

    if args.self_test:
        if pd is None:
            print("SKIP: pandas not available for self-test")
            return 0
        df_empty = pd.DataFrame(); df_full  = pd.DataFrame({"a":[1]})
        assert _pick_df(df_full, df_empty) is df_full
        assert _pick_df(df_empty, df_full) is df_full
        print("SELF-TEST OK")
        return 0

    if args.proxy: configure_proxy(args.proxy)
    if not HAS_YF:
        _eprint('[ERROR] 未安装 yfinance，无法抓取。请先 pip install -U yfinance'); return 2

    # symbols 来源
    syms:List[str]=[]
    if args.symbols:
        syms=[s.strip().upper().replace('.', '-') for s in args.symbols.split(',') if s.strip()]
    elif args.from_file and os.path.exists(args.from_file):
        with open(args.from_file,'r',encoding='utf-8') as f:
            syms=[line.strip().upper().replace('.', '-') for line in f if line.strip()]
    else:
        syms=get_sp500_symbols()
    if args.dump_sp500:
        print('count=', len(syms))
        for s in syms[:10]: print(s)
        return 0
    if not syms:
        _eprint('[ERROR] 没有待抓取的代码。'); return 2

    # 分片（可多进程/多机分摊）
    shard=args.shard; total=args.of
    if total < 1: total = 1
    if shard < 1: shard = 1
    if shard > total: shard = total
    if total > 1:
        syms = [s for i, s in enumerate(syms) if (i % total) == (shard - 1)]
    if args.shuffle:
        random.shuffle(syms)

    # DB & 简单过期判断
    conn=db_open(args.db)
    existing=db_get_existing(conn) if args.resume else {}
    now=time.time(); stale_sec=args.stale_days*86400

    # 限流器
    limiter = QPSLimiter(args.qps, args.sleep)

    # 并发抓取（轻量）
    from concurrent.futures import ThreadPoolExecutor, as_completed
    prog=Progress(len(syms), enabled=(not args.no_progress))
    ok=0; fail=0

    RATE_LIMIT_PATTERNS = ('Too Many Requests', 'Rate Limited', '429')

    def _one(sym:str)->Optional[Dict[str,Any]]:
        # 跳过“未过期”的票
        if args.resume and sym in existing and (now-existing[sym]) < stale_sec:
            return None  # 视为已完成
        # 重试 + 限流 + 指数退避
        delay_base = 4.0
        for i in range(1, args.retries+1):
            limiter.wait()
            try:
                res=fetch_metrics(sym)
                return res
            except Exception as e:
                msg = str(e)
                is_rate = any(pat.lower() in msg.lower() for pat in RATE_LIMIT_PATTERNS)
                if i >= args.retries:
                    raise
                if is_rate:
                    backoff = min(120.0, delay_base * (2 ** (i-1))) + random.uniform(0.0, 1.5)
                    _eprint(f"[429] {sym} 第{i}次被限流，退避 {backoff:.1f}s 后重试…")
                    time.sleep(backoff)
                else:
                    time.sleep(min(2.0*i, 5.0))
        return None

    with ThreadPoolExecutor(max_workers=max(1,args.max_workers)) as ex:
        futs={ex.submit(_one, s): s for s in syms}
        for fu in as_completed(futs):
            sym=futs[fu]
            try:
                row=fu.result()
                if row is not None:
                    db_upsert(conn,row)
                    ok+=1; prog.upd(True)
                else:
                    prog.upd(True)
            except Exception as e:
                fail+=1; prog.upd(False)
                _eprint(f"[WARN] 抓取失败（已跳过）：{sym} -> {e}")
    prog.close(); conn.commit(); conn.close()
    _eprint(f"[SUMMARY] 抓取完成 ok={ok}, fail={fail}, total={len(syms)}，数据库：{args.db}")
    return 0

if __name__=='__main__':
    sys.exit(fetch_main())
