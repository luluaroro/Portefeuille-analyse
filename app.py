# app.py
import uuid
from pathlib import Path
import os
import math
import json
import datetime as dt
import hashlib
import re
from typing import Optional, Dict, Any, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import httpx
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="Portefeuille & Analyse", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
TX_PATH = os.path.join(DATA_DIR, "transactions.csv")
ISIN_CACHE_PATH = os.path.join(DATA_DIR, "isin_fr_cache.json")

FR_MAPPING_REFRESH_SECONDS = int(os.getenv("FR_MAPPING_REFRESH_SECONDS", "3600"))  # 1h
DEFAULT_REFRESH_SECONDS = 10  # cache yfinance (1s)
PRICE_HISTORY_TTL = 60
LAST_PRICE_TTL = 10
STATIC_DIR = Path(__file__).resolve().parent / "static"
LW_JS_PATH = STATIC_DIR / "lightweight-charts.standalone.production.js"
# ---------------------------
# Helpers: storage
# ---------------------------
def load_isin_cache() -> dict:
    if not os.path.exists(ISIN_CACHE_PATH):
        return {}
    try:
        with open(ISIN_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_isin_cache(cache: dict) -> None:
    try:
        with open(ISIN_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def yahoo_suffix_from_exchange_label(exch_label: str) -> str:
    if not exch_label:
        return ""
    s = exch_label.lower()
    if "euronext paris" in s:
        return ".PA"
    if "euronext amsterdam" in s:
        return ".AS"
    if "xetra" in s or "frankfurt" in s:
        return ".DE"
    if "london stock exchange" in s:
        return ".L"
    if "six swiss exchange" in s:
        return ".SW"
    return ""

@st.cache_data(ttl=FR_MAPPING_REFRESH_SECONDS, show_spinner=False)
def build_fr_isin_mapping_from_wikidata(limit: int = 20000) -> dict:
    sparql = f"""
    SELECT ?isin ?ticker ?name ?exchLabel WHERE {{
      ?item wdt:P946 ?isin.
      FILTER(STRSTARTS(?isin, "FR"))
      OPTIONAL {{ ?item wdt:P249 ?ticker. }}
      OPTIONAL {{ ?item rdfs:label ?name FILTER(lang(?name)="fr"). }}
      OPTIONAL {{ ?item wdt:P414 ?exch. }}
      OPTIONAL {{ ?exch rdfs:label ?exchLabel FILTER(lang(?exchLabel)="en"). }}
    }} LIMIT {limit}
    """

    url = "https://query.wikidata.org/sparql"
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "streamlit-app/1.0 (ISIN resolver)"
    }

    mapping: dict = {}
    with httpx.Client(timeout=30.0) as client:
        r = client.get(url, params={"query": sparql, "format": "json"}, headers=headers)
        r.raise_for_status()
        data = r.json()

    for b in data.get("results", {}).get("bindings", []):
        isin = (b.get("isin", {}) or {}).get("value")
        tk = (b.get("ticker", {}) or {}).get("value")
        name = (b.get("name", {}) or {}).get("value") or ""
        exch = (b.get("exchLabel", {}) or {}).get("value") or ""

        if not isin or not tk:
            continue

        suffix = yahoo_suffix_from_exchange_label(exch)
        yahoo_ticker = tk if "." in tk else (tk + suffix if suffix else tk)

        mapping.setdefault(isin, [])
        mapping[isin].append({"ticker": yahoo_ticker, "name": name, "exch": exch})

    # dédoublonnage
    for isin, lst in mapping.items():
        seen = set()
        uniq = []
        for x in lst:
            if x["ticker"] not in seen:
                uniq.append(x)
                seen.add(x["ticker"])
        mapping[isin] = uniq

    return mapping

def resolve_isin_to_tickers(isin: str) -> list[dict]:
    isin = (isin or "").strip().upper()
    if not isin:
        return []

    local_cache = load_isin_cache()
    if isin in local_cache and local_cache[isin]:
        return local_cache[isin]

    results: list[dict] = []

    # 1) Wikidata bulk FR
    try:
        fr_map = build_fr_isin_mapping_from_wikidata()
        if isin in fr_map and fr_map[isin]:
            local_cache[isin] = fr_map[isin]
            save_isin_cache(local_cache)
            return fr_map[isin]
    except Exception:
        pass

    # 1bis) Wikidata fallback (par ISIN)
    try:
        sparql = f"""
        SELECT ?ticker ?name ?exchLabel WHERE {{
          ?item wdt:P946 "{isin}".
          OPTIONAL {{ ?item wdt:P249 ?ticker. }}
          OPTIONAL {{ ?item rdfs:label ?name FILTER(lang(?name)="fr"). }}
          OPTIONAL {{ ?item wdt:P414 ?exch. }}
          OPTIONAL {{ ?exch rdfs:label ?exchLabel FILTER(lang(?exchLabel)="en"). }}
        }} LIMIT 10
        """
        url = "https://query.wikidata.org/sparql"
        headers = {"Accept": "application/sparql-results+json", "User-Agent": "streamlit-app/1.0"}
        with httpx.Client(timeout=20.0) as client:
            r = client.get(url, params={"query": sparql, "format": "json"}, headers=headers)
            r.raise_for_status()
            data = r.json()

        for b in data.get("results", {}).get("bindings", []):
            tk = (b.get("ticker", {}) or {}).get("value")
            name = (b.get("name", {}) or {}).get("value") or ""
            exch = (b.get("exchLabel", {}) or {}).get("value") or ""
            if tk:
                suffix = yahoo_suffix_from_exchange_label(exch)
                yahoo_tk = tk if "." in tk else (tk + suffix if suffix else tk)
                results.append({"ticker": yahoo_tk, "name": name, "exch": exch})
    except Exception:
        pass

    # 2) OpenFIGI (optionnel)
    of_key = os.getenv("OPENFIGI_API_KEY")
    if of_key:
        try:
            url = "https://api.openfigi.com/v3/mapping"
            headers = {"X-OPENFIGI-APIKEY": of_key, "Content-Type": "application/json"}
            payload = [{"idType": "ID_ISIN", "idValue": isin}]
            with httpx.Client(timeout=20.0) as client:
                r = client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()

            data0 = data[0] if isinstance(data, list) and len(data) > 0 else {}
            for item in (data0.get("data") or []):
                results.append({
                    "ticker": item.get("ticker"),
                    "name": item.get("name"),
                    "exch": item.get("exchCode"),
                })
        except Exception:
            pass

    # 3) FMP fallback (optionnel)
    if not results:
        fmp_key = os.getenv("FMP_API_KEY")
        if fmp_key:
            try:
                url = "https://financialmodelingprep.com/stable/search-isin"
                with httpx.Client(timeout=20.0) as client:
                    r = client.get(url, params={"isin": isin, "apikey": fmp_key})
                    r.raise_for_status()
                    data = r.json()

                if isinstance(data, list):
                    for item in data:
                        results.append({
                            "ticker": item.get("symbol"),
                            "name": item.get("companyName"),
                            "exch": item.get("exchange"),
                        })
            except Exception:
                pass

    # Nettoyage + cache
    results = [x for x in results if x.get("ticker")]
    seen = set()
    uniq = []
    for x in results:
        if x["ticker"] not in seen:
            uniq.append(x)
            seen.add(x["ticker"])

    if uniq:
        local_cache = load_isin_cache()
        local_cache[isin] = uniq
        save_isin_cache(local_cache)

    return uniq

def is_isin(x: str) -> bool:
    if not x:
        return False
    x = x.strip().upper()
    return bool(re.fullmatch(r"[A-Z]{2}[A-Z0-9]{10}", x))

def load_transactions() -> pd.DataFrame:
    if not os.path.exists(TX_PATH):
        df = pd.DataFrame(columns=["date", "ticker", "asset_name", "type", "quantity", "price", "currency", "fees"])
        df.to_csv(TX_PATH, index=False)
        return df
    df = pd.read_csv(TX_PATH)
    if len(df) == 0:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["fees"] = pd.to_numeric(df["fees"], errors="coerce").fillna(0.0)
    return df

def save_transactions(df: pd.DataFrame) -> None:
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"]).dt.strftime("%Y-%m-%d")
    df2.to_csv(TX_PATH, index=False)

# ---------------------------
# Market data
# ---------------------------
@st.cache_data(ttl=LAST_PRICE_TTL, show_spinner=False)
def get_last_price(ticker: str) -> Tuple[Optional[float], Optional[str]]:
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        price = info.get("last_price", None)
        ccy = info.get("currency", None)
        if price is None:
            hist = t.history(period="5d")
            if hist is not None and len(hist) > 0:
                price = float(hist["Close"].iloc[-1])
        return (float(price) if price is not None and not math.isnan(float(price)) else None, ccy)
    except Exception:
        return (None, None)

@st.cache_data(ttl=PRICE_HISTORY_TTL, show_spinner=False)
def get_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    t = yf.Ticker(ticker)
    if period == "1d":
        hist = t.history(period="1d", interval="1m")
    else:
        hist = t.history(period=period)

    if hist is None or len(hist) == 0:
        return pd.DataFrame()

    hist = hist.reset_index()
    if "Date" not in hist.columns and "Datetime" in hist.columns:
        hist = hist.rename(columns={"Datetime": "Date"})
    return hist[["Date", "Close"]].dropna()


@st.cache_data(show_spinner=False)
def load_lightweight_js() -> str:
    """
    Charge le bundle Lightweight Charts depuis /static.
    On le met en cache Streamlit car le fichier est lourd.
    """
    try:
        if not LW_JS_PATH.exists():
            return ""
        return LW_JS_PATH.read_text(encoding="utf-8")
    except Exception:
        return ""
    
def render_lightweight_area_chart(hist: pd.DataFrame, height: int = 420, currency: str = "", title: str = ""):
    if hist is None or hist.empty:
        st.info("Historique vide.")
        return

    df = hist.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"])

    data = [{"time": int(ts.timestamp()), "value": float(val)} for ts, val in zip(df["Date"], df["Close"])]
    if len(data) < 2:
        st.info("Pas assez de points pour afficher un graphique.")
        return

    # Si tu utilises le fichier local
    lw_js = load_lightweight_js()
    st.write("LW_JS exists:", LW_JS_PATH.exists())
    st.write("LW_JS path:", str(LW_JS_PATH))
    st.write("LW_JS length:", len(lw_js))
    st.write("LW_JS head:", lw_js[:80])

    if not lw_js:
        st.error("Le fichier lightweight-charts.standalone.production.js est introuvable (dossier static/).")
        return

    up = data[-1]["value"] >= data[0]["value"]
    line = "#22c55e" if up else "#ef4444"
    top = "rgba(34,197,94,0.25)" if up else "rgba(239,68,68,0.25)"
    bottom = "rgba(34,197,94,0.02)" if up else "rgba(239,68,68,0.02)"

    chart_id = f"tvchart_{uuid.uuid4().hex}"

    series_opts = {
        "lineColor": line,
        "topColor": top,
        "bottomColor": bottom,
        "lineWidth": 2,
    }

    html = f"""
<div style="color:rgba(255,255,255,0.85); font-size:14px; font-weight:600; margin:6px 0 10px 0;">
  {title}
</div>

<div id="dbg" style="font-family:monospace; font-size:12px; color:#9ae6b4; margin-bottom:8px;"></div>
<div id="{chart_id}" style="width:100%; height:{height}px; border:1px solid rgba(255,255,255,0.08);"></div>

<script>
  const dbg = document.getElementById("dbg");
  function log(msg) {{
    if (dbg) dbg.innerText += msg + "\\n";
    console.log(msg);
  }}

  window.onerror = function(message, source, lineno, colno, error) {{
    log("❌ JS ERROR: " + message + " @ " + lineno + ":" + colno);
  }};

  log("Iframe ready ✅");
  log("LightweightCharts typeof (before inject) = " + (typeof window.LightweightCharts));
</script>

<script>
{lw_js}
</script>

<script>
(function() {{
  const el = document.getElementById("{chart_id}");
  if (!el) {{
    log("❌ chart container introuvable");
    return;
  }}

  const seriesData = {json.dumps(data)};

  function tryCreateChart(attempt=0) {{
    try {{
      const LC = window.LightweightCharts;
      log("Attempt " + attempt + " | LC=" + (LC ? "OK" : "NO") + " | width=" + el.clientWidth);

      if (!LC) {{
        const s = document.createElement("script");
        s.src = "https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js";
        s.onload = () => {{
          log("✅ CDN loaded");
          setTimeout(() => tryCreateChart(attempt+1), 100);
        }};
        s.onerror = () => log("❌ CDN failed to load");
        document.head.appendChild(s);
        return;
      }}

      if ((el.clientWidth || 0) < 50 && attempt < 10) {{
        setTimeout(() => tryCreateChart(attempt+1), 150);
        return;
      }}

      el.innerHTML = "";
      const chart = LC.createChart(el, {{
        width: el.clientWidth || 600,
        height: {height},
        layout: {{
          background: {{ type: 'solid', color: '#0b0f14' }},
          textColor: 'rgba(255,255,255,0.85)',
        }},
        grid: {{
          vertLines: {{ visible: false }},
          horzLines: {{ visible: false }},
        }},
        rightPriceScale: {{ borderVisible: false }},
        timeScale: {{ borderVisible: false, timeVisible: true, secondsVisible: false }},
      }});

      const series = chart.addSeries(LC.AreaSeries, {json.dumps(series_opts)});
      series.setData(seriesData);
      chart.timeScale().fitContent();
      log("✅ Chart created + data set (" + seriesData.length + " pts)");

      if (typeof ResizeObserver !== "undefined") {{
        const ro = new ResizeObserver(() => {{
          chart.applyOptions({{ width: el.clientWidth || 600 }});
        }});
        ro.observe(el);
        log("✅ ResizeObserver ON");
      }} else {{
        log("⚠️ ResizeObserver absent");
      }}

    }} catch (e) {{
      log("❌ Exception: " + (e && e.message ? e.message : e));
    }}
  }}

  setTimeout(() => tryCreateChart(0), 150);
}})();
</script>
"""
    components.html(html, height=height, scrolling=False)

# ---------------------------
# Financial analysis (heuristics)
# ---------------------------
def detect_asset_type(info: dict) -> str:
    qt = (info.get("quoteType") or "").upper()
    it = (info.get("instrumentType") or "").upper()
    name = (info.get("longName") or info.get("shortName") or "").upper()

    # 1) Signaux explicites
    if "ETF" in qt or "ETF" in it:
        return "ETF"
    if "BOND" in qt or "BOND" in it or "DEBT" in it or "FIXED" in it:
        return "OBLIGATION"
    if "EQUITY" in qt or "STOCK" in qt or "EQUITY" in it:
        return "ACTION"

    # 2) Signaux “fonds / ETF”
    etf_signals = [
        info.get("fundFamily"),
        info.get("totalAssets"),
        info.get("annualReportExpenseRatio"),
        info.get("category"),
    ]
    if any(x is not None for x in etf_signals) or "UCITS" in name or "ETF" in name:
        return "ETF"

    # 3) Signaux obligations
    bond_signals = [
        info.get("maturityDate"),
        info.get("couponRate"),
        info.get("yield"),
    ]
    if any(x is not None for x in bond_signals) or "BOND" in name or "TREASURY" in name:
        return "OBLIGATION"

    return "AUTRE"

def safe_div(a: float, b: float) -> Optional[float]:
    try:
        if b is None or b == 0 or (isinstance(b, float) and np.isnan(b)):
            return None
        return float(a) / float(b)
    except Exception:
        return None

def clamp(x: float, lo=0.0, hi=100.0) -> float:
    return float(max(lo, min(hi, x)))

def score_from_ranges(value: Optional[float], good: Tuple[float, float], ok: Tuple[float, float], higher_is_better=True) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 50.0
    v = float(value)

    def in_range(v, r):
        return v >= r[0] and v <= r[1]

    if higher_is_better:
        if in_range(v, good):
            return clamp(85 + 15 * (v - good[0]) / (good[1] - good[0] + 1e-9))
        if in_range(v, ok):
            return clamp(60 + 25 * (v - ok[0]) / (ok[1] - ok[0] + 1e-9))
        if v < ok[0]:
            return clamp(60 * (v / (ok[0] + 1e-9)))
        return 100.0
    else:
        if in_range(v, good):
            return clamp(85 + 15 * (good[1] - v) / (good[1] - good[0] + 1e-9))
        if in_range(v, ok):
            return clamp(60 + 25 * (ok[1] - v) / (ok[1] - ok[0] + 1e-9))
        if v > ok[1]:
            return clamp(60 * (ok[1] / (v + 1e-9)))
        return 100.0

@st.cache_data(ttl=6*60*60, show_spinner=False)
def fetch_financials(ticker: str) -> Dict[str, Any]:
    t = yf.Ticker(ticker)
    fin = {}
    try:
        fin["info"] = t.get_info()
    except Exception:
        fin["info"] = {}

    try:
        fin["income_stmt"] = t.income_stmt
    except Exception:
        fin["income_stmt"] = pd.DataFrame()

    try:
        fin["cashflow"] = t.cashflow
    except Exception:
        fin["cashflow"] = pd.DataFrame()

    try:
        fin["balance_sheet"] = t.balance_sheet
    except Exception:
        fin["balance_sheet"] = pd.DataFrame()

    return fin

def pick_latest(df: pd.DataFrame, row_names: list) -> Optional[float]:
    if df is None or len(df) == 0:
        return None
    for r in row_names:
        if r in df.index:
            s = df.loc[r].dropna()
            if len(s) > 0:
                return float(s.iloc[0])
    return None

def compute_metrics(fin: Dict[str, Any]) -> Dict[str, Optional[float]]:
    inc = fin.get("income_stmt", pd.DataFrame())
    cf = fin.get("cashflow", pd.DataFrame())
    bs = fin.get("balance_sheet", pd.DataFrame())
    info = fin.get("info", {})

    revenue = pick_latest(inc, ["Total Revenue", "TotalRevenue"])
    net_income = pick_latest(inc, ["Net Income", "NetIncome"])
    gross_profit = pick_latest(inc, ["Gross Profit", "GrossProfit"])

    fcf = pick_latest(cf, ["Free Cash Flow", "FreeCashFlow"])
    cfo = pick_latest(cf, ["Total Cash From Operating Activities", "Operating Cash Flow", "OperatingCashFlow"])
    capex = pick_latest(cf, ["Capital Expenditures", "CapitalExpenditures"])
    if fcf is None and cfo is not None and capex is not None:
        fcf = cfo - capex

    gross_margin = safe_div(gross_profit, revenue) if gross_profit is not None and revenue is not None else None
    net_margin = safe_div(net_income, revenue) if net_income is not None and revenue is not None else None
    fcf_margin = safe_div(fcf, revenue) if fcf is not None and revenue is not None else None

    total_equity = pick_latest(bs, ["Total Stockholder Equity", "TotalStockholderEquity", "Stockholders Equity"])
    roe = safe_div(net_income, total_equity) if net_income is not None and total_equity is not None else None

    total_debt = pick_latest(bs, ["Total Debt", "TotalDebt"])
    if total_debt is None:
        short_debt = pick_latest(bs, ["Short Long Term Debt", "ShortLongTermDebt", "Short Term Debt", "ShortTermDebt"])
        long_debt = pick_latest(bs, ["Long Term Debt", "LongTermDebt"])
        if short_debt is not None or long_debt is not None:
            total_debt = (short_debt or 0.0) + (long_debt or 0.0)

    ebitda = info.get("ebitda", None)
    debt_to_ebitda = safe_div(total_debt, ebitda) if total_debt is not None and ebitda is not None else None

    pe = info.get("trailingPE", None) or info.get("forwardPE", None)
    market_cap = info.get("marketCap", None)
    p_fcf = safe_div(market_cap, fcf) if market_cap is not None and fcf is not None and fcf > 0 else None

    rev_cagr = None
    fcf_cagr = None
    if inc is not None and len(inc) > 0 and "Total Revenue" in inc.index:
        rev_series = inc.loc["Total Revenue"].dropna()
        if len(rev_series) >= 3:
            newest = float(rev_series.iloc[0])
            oldest = float(rev_series.iloc[min(2, len(rev_series)-1)])
            years = 2.0
            if oldest > 0 and newest > 0:
                rev_cagr = (newest / oldest) ** (1/years) - 1

    return dict(
        revenue=revenue,
        net_income=net_income,
        fcf=fcf,
        gross_margin=gross_margin,
        net_margin=net_margin,
        fcf_margin=fcf_margin,
        roe=roe,
        debt_to_ebitda=debt_to_ebitda,
        pe=pe,
        p_fcf=p_fcf,
        rev_cagr=rev_cagr,
        fcf_cagr=fcf_cagr,
        market_cap=market_cap,
        currency=info.get("currency", None),
        long_name=info.get("longName", None) or info.get("shortName", None),
        sector=info.get("sector", None),
        industry=info.get("industry", None),
    )

def compute_scores(m: Dict[str, Optional[float]]) -> Dict[str, float]:
    growth = max(m.get("rev_cagr") or 0.0, m.get("fcf_cagr") or 0.0)
    growth_score = score_from_ranges(growth, good=(0.10, 0.40), ok=(0.03, 0.10), higher_is_better=True)

    lev = m.get("debt_to_ebitda")
    debt_score = score_from_ranges(lev, good=(0.0, 2.0), ok=(2.0, 4.0), higher_is_better=False)

    nm = m.get("net_margin")
    gm = m.get("gross_margin")
    fcfm = m.get("fcf_margin")
    prof_score = float(np.mean([
        score_from_ranges(gm, good=(0.40, 0.80), ok=(0.25, 0.40), higher_is_better=True),
        score_from_ranges(nm, good=(0.15, 0.35), ok=(0.05, 0.15), higher_is_better=True),
        score_from_ranges(fcfm, good=(0.10, 0.30), ok=(0.03, 0.10), higher_is_better=True),
    ]))

    roe = m.get("roe")
    rent_score = score_from_ranges(roe, good=(0.15, 0.35), ok=(0.08, 0.15), higher_is_better=True)

    pe = m.get("pe")
    pfcf = m.get("p_fcf")
    val_score = float(np.mean([
        score_from_ranges(pe, good=(8.0, 18.0), ok=(18.0, 30.0), higher_is_better=False),
        score_from_ranges(pfcf, good=(10.0, 20.0), ok=(20.0, 35.0), higher_is_better=False),
    ]))

    overall = float(np.mean([growth_score, debt_score, prof_score, rent_score, val_score]))
    return dict(
        croissance=growth_score,
        endettement=debt_score,
        profitabilite=prof_score,
        rentabilite=rent_score,
        valorisation=val_score,
        score_global=overall,
    )

def rule_based_comment(scores: Dict[str, float], metrics: Dict[str, Optional[float]]) -> str:
    g = scores["score_global"]
    parts = []
    if g >= 75:
        parts.append("Profil global **plutôt solide** au vu des métriques disponibles.")
    elif g >= 60:
        parts.append("Profil global **correct**, mais pas irréprochable.")
    else:
        parts.append("Profil global **fragile / incertain** selon les données récupérées.")

    best = max((k for k in scores.keys() if k != "score_global"), key=lambda k: scores[k])
    worst = min((k for k in scores.keys() if k != "score_global"), key=lambda k: scores[k])
    parts.append(f"Point fort: **{best}** ({scores[best]:.0f}/100).")
    parts.append(f"Point faible: **{worst}** ({scores[worst]:.0f}/100).")

    if g >= 75 and scores["valorisation"] >= 55:
        parts.append("Lecture rapide: **intéressant**, à creuser (risques sectoriels, guidance, concurrence).")
    elif g >= 60:
        parts.append("Lecture rapide: **watchlist** (attendre un meilleur prix ou un signal fondamental).")
    else:
        parts.append("Lecture rapide: **prudence** (marge/leverage/croissance à vérifier).")

    return " ".join(parts)

# ---------------------------
# IA (OpenAI + Ollama)
# ---------------------------
def openai_investment_report(asset_type: str, ticker: str, isin: str, scores: dict, metrics: dict, extra: dict) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    if asset_type == "ACTION":
        focus = "- Action : marges/ROE/FCF/dette-EBITDA/croissance/valorisation/risques. Ne rien inventer."
    elif asset_type == "ETF":
        focus = "- ETF : frais/encours/perf/volatilité/drawdown/cohérence. Ne pas parler de ROE/PER."
    elif asset_type == "OBLIGATION":
        focus = "- Obligation : rendement/coupon/stabilité/échéance/risques taux-crédit. Ne pas parler de PER."
    else:
        focus = "- Analyse prudente selon les données disponibles."

    prompt = f"""
Tu es un analyste financier prudent. Réponds en français, structuré, sans inventer de données.
Verdict final obligatoire: "FAVORABLE" ou "DEFAVORABLE".

Type: {asset_type}
Ticker: {ticker}
ISIN: {isin}

Scores: {json.dumps(scores, ensure_ascii=False)}
Métriques: {json.dumps(metrics, ensure_ascii=False)}
Extra: {json.dumps(extra, ensure_ascii=False)}

Consignes: {focus}

Format:
1) Résumé (3 lignes max)
2) Forces
3) Faiblesses / risques
4) Points à vérifier
5) Verdict: FAVORABLE ou DEFAVORABLE + justification
Termine EXACTEMENT par: "Ce n’est pas un conseil financier."
"""

    payload = {
    "model": model,
    "messages": [
        {
            "role": "system",
            "content": (
                "Tu es un analyste financier senior, expérimenté en marchés financiers. "
                "Réponds en français. "
                "Tu analyses UNIQUEMENT à partir des données fournies. "
                "Si une donnée est absente ou incertaine, écris clairement : 'non disponible'. "
                "Tu dois produire une analyse rigoureuse, structurée et professionnelle. "
                "Tu dois conclure par un verdict clair : FAVORABLE ou DEFAVORABLE, "
                "en justifiant ce verdict uniquement par les données analysées. "
                "Tu n’inventes jamais de chiffres."
            ),
        },
        {"role": "user", "content": prompt},
    ],
    "temperature": 0.2,
    "max_tokens": 750,
}

    try:
        with httpx.Client(timeout=35.0) as client:
            r = client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None

def ollama_chat(prompt: str, model: str = "qwen2.5:7b") -> Optional[str]:
    """
    Envoie un prompt à Ollama (local) et renvoie la réponse.
    Nécessite Ollama actif sur http://localhost:11434
    Optimisé pour analyse financière experte.
    """
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.15,
                "top_p": 0.9,
                "num_ctx": 8192,
                "num_predict": 700
            }
        }

        with httpx.Client(timeout=90.0) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            return (data.get("response") or "").strip()

    except Exception as e:
        return f"Ollama error: {e}"

def ollama_investment_report(asset_type: str, ticker: str, isin: str, scores: dict, metrics: dict, extra: dict, model: str = "mistral") -> Optional[str]:
    if asset_type == "ACTION":
        focus = "Analyse action : marges, ROE, FCF, dette/EBITDA, croissance, valorisation, risques."
    elif asset_type == "ETF":
        focus = "Analyse ETF : frais, encours, performance, volatilité, drawdown, cohérence."
    elif asset_type == "OBLIGATION":
        focus = "Analyse obligation : rendement, coupon, stabilité, échéance, risques taux/crédit."
    else:
        focus = "Analyse prudente basée sur les données disponibles."

    prompt = f"""
    Tu es un analyste financier senior, expert en analyse de marché et analyse financière.
    Réponds en français.
    Tu analyses uniquement à partir des données fournies.
    Si une donnée est absente, indique clairement : "non disponible".
    N’invente aucun chiffre ou information.
    Ton analyse doit être rigoureuse, structurée et professionnelle.
    Tu dois conclure par un verdict clair : FAVORABLE ou DEFAVORABLE.

Type d’actif: {asset_type}
Ticker: {ticker}
ISIN: {isin}

Scores:
{json.dumps(scores, ensure_ascii=False)}

Métriques:
{json.dumps(metrics, ensure_ascii=False)}

Infos complémentaires:
{json.dumps(extra, ensure_ascii=False)}

Consignes:
{focus}

Format obligatoire:
1) Résumé (3 lignes max)
2) Forces
3) Faiblesses / risques
4) Verdict final : FAVORABLE ou DEFAVORABLE
Termine EXACTEMENT par : "Ce n’est pas un conseil financier."
"""
    return ollama_chat(prompt, model=model)

# ---------------------------
# Portfolio computations
# ---------------------------
def compute_positions(tx: pd.DataFrame) -> pd.DataFrame:
    if tx is None or len(tx) == 0:
        return pd.DataFrame(columns=["ticker", "asset_name", "quantity", "avg_cost", "cost_basis"])
    df = tx.copy()
    df["signed_qty"] = np.where(df["type"].str.lower().str.strip() == "sell", -df["quantity"], df["quantity"])
    df["signed_value"] = np.where(
        df["type"].str.lower().str.strip() == "sell",
        -(df["quantity"] * df["price"] - df["fees"]),
        (df["quantity"] * df["price"] + df["fees"])
    )
    grouped = df.groupby(["ticker", "asset_name"], dropna=False).agg(
        quantity=("signed_qty", "sum"),
        cost_basis=("signed_value", "sum"),
    ).reset_index()
    grouped["avg_cost"] = grouped.apply(lambda r: (r["cost_basis"] / r["quantity"]) if r["quantity"] != 0 else np.nan, axis=1)
    return grouped

# ---------------------------
# UI
# ---------------------------
st.title("Outil de gestion & d’analyse de portefeuille")
tabs = st.tabs(["📒 Portefeuille", "🔎 Analyse d’un investissement", "⚙️ Paramètres"])

# ---- Portfolio tab
with tabs[0]:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Ajouter une transaction")
        tx = load_transactions()

        with st.form("add_tx", clear_on_submit=True):
            c1, c2 = st.columns([1, 1])
            date = c1.date_input("Date", value=dt.date.today())
            tx_type = c2.selectbox("Type", ["Buy", "Sell"])
            ticker = st.text_input("Ticker (ex: AAPL, MC.PA, CW8.PA, VWCE.DE)", value="").strip()
            asset_name = st.text_input("Nom (optionnel)", value="")
            c3, c4, c5 = st.columns([1, 1, 1])
            qty = c3.number_input("Quantité", min_value=0.0, value=1.0, step=1.0)
            price = c4.number_input("Prix unitaire", min_value=0.0, value=0.0, step=0.01)
            fees = c5.number_input("Frais", min_value=0.0, value=0.0, step=0.01)
            currency = st.text_input("Devise (optionnel)", value="")
            submitted = st.form_submit_button("Enregistrer")
            if submitted:
                if not ticker:
                    st.error("Ticker requis.")
                else:
                    new = pd.DataFrame([{
                        "date": date,
                        "ticker": ticker,
                        "asset_name": asset_name,
                        "type": tx_type,
                        "quantity": qty,
                        "price": price,
                        "currency": currency,
                        "fees": fees
                    }])
                    tx2 = pd.concat([tx, new], ignore_index=True)
                    save_transactions(tx2)
                    st.success("Transaction enregistrée.")

        st.divider()
        st.subheader("Transactions")
        tx = load_transactions()
        st.dataframe(tx.sort_values(["date"], ascending=False), use_container_width=True)

    with right:
        st.subheader("Positions & P/L (rafraîchi)")
        tx = load_transactions()
        pos = compute_positions(tx)

        if len(pos) == 0:
            st.info("Aucune position pour le moment.")
        else:
            rows = []
            for _, r in pos.iterrows():
                tk = str(r["ticker"])
                qty = float(r["quantity"])
                last, ccy = get_last_price(tk)
                if last is None:
                    last = np.nan
                mv = qty * last if not np.isnan(last) else np.nan
                cost = float(r["cost_basis"])
                pl = mv - cost if not np.isnan(mv) else np.nan
                pl_pct = (pl / cost) if cost != 0 and not np.isnan(pl) else np.nan

                rows.append({
                    "Ticker": tk,
                    "Nom": r.get("asset_name", ""),
                    "Quantité": qty,
                    "PRU": r.get("avg_cost", np.nan),
                    "Dernier cours": last,
                    "Valeur": mv,
                    "Coût": cost,
                    "P/L": pl,
                    "P/L %": pl_pct,
                    "Devise": ccy or "",
                })

            pf = pd.DataFrame(rows)
            st.dataframe(pf, use_container_width=True)

            total_value = float(np.nansum(pf["Valeur"].values))
            total_cost = float(np.nansum(pf["Coût"].values))
            total_pl = total_value - total_cost
            st.metric("Valeur totale (somme)", f"{total_value:,.2f}", delta=f"{total_pl:,.2f}")

        st.caption(f"Cache yfinance: {DEFAULT_REFRESH_SECONDS}s.")

# ---- Analysis tab
with tabs[1]:
    st_autorefresh(interval=2000, limit=None, key="analysis_refresh")
    st.caption("Auto-refresh : 2s")
    st.subheader("Analyser un investissement (ISIN → ticker)")

    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    isin = c1.text_input("ISIN (ex: US0378331005 / FR0000120321)", value="").strip()
    ticker_manual = c2.text_input("Ticker (optionnel si tu le connais)", value="").strip()
    period = c3.selectbox("Période graphique", ["1d", "3mo", "6mo", "1y", "3y", "5y"], index=3)
    ai_provider = c4.selectbox("IA d’analyse", ["Aucune", "Ollama (gratuit)", "OpenAI"], index=0)

    ticker = ticker_manual

    if (not isin) and is_isin(ticker):
        isin = ticker
        ticker = ""
        st.info("ISIN détecté dans le champ Ticker → déplacé automatiquement dans le champ ISIN.")

    if is_isin(ticker):
        st.error("Le champ Ticker contient un ISIN. Mets l’ISIN dans le champ ISIN, ou laisse ticker vide.")
        st.stop()

    if (not ticker) and isin:
        cands = resolve_isin_to_tickers(isin)
        if not cands:
            st.error("ISIN introuvable. Vérifie l’ISIN et ta variable FMP_API_KEY.")
            st.stop()

        choice = st.selectbox(
            "Sélectionne l’instrument trouvé",
            cands,
            format_func=lambda x: f"{x.get('ticker','?')} — {x.get('name','')} ({x.get('exch','')})"
        )
        ticker = choice.get("ticker", "").strip()

    if not ticker:
        st.info("Entre un ISIN (recommandé) ou un ticker, puis lance l’analyse.")
        st.stop()

    # ✅ Fix: ne refetch l'historique que si ticker/période change
    chart_key = f"{ticker}|{period}"
    if st.session_state.get("chart_key") != chart_key:
        hist = get_price_history(ticker, period=period)
        st.session_state["chart_key"] = chart_key
        st.session_state["chart_hist"] = hist
    else:
        hist = st.session_state.get("chart_hist", pd.DataFrame())

    if hist is None or len(hist) == 0:
        st.warning("Historique de prix indisponible pour ce ticker.")
    else:
        last, ccy = get_last_price(ticker)
        render_lightweight_area_chart(hist, height=420, currency=ccy or "", title=f"{ticker} — cours")
        components.html("<div style='color:lime; font-weight:700'>TEST COMPONENTS OK</div>", height=40)
    info = {}
    try:
        info = yf.Ticker(ticker).get_info()
    except Exception:
        info = {}

    asset_type = detect_asset_type(info)
    extra = {
        "quoteType": info.get("quoteType"),
        "instrumentType": info.get("instrumentType"),
        "longName": info.get("longName") or info.get("shortName"),
        "currency": info.get("currency"),
        # ETF-ish
        "fundFamily": info.get("fundFamily"),
        "category": info.get("category"),
        "totalAssets": info.get("totalAssets"),
        "annualReportExpenseRatio": info.get("annualReportExpenseRatio"),
        # Bond-ish
        "maturityDate": info.get("maturityDate"),
        "couponRate": info.get("couponRate"),
        "yield": info.get("yield"),
    }

    if asset_type == "ACTION":
        fin = fetch_financials(ticker)
        m = compute_metrics(fin)
        s = compute_scores(m)
    else:
        m = {"note": "Analyse limitée (seulement ACTION implémenté ici)."}
        s = {"score_global": 50.0}

    st.caption(f"Type détecté: **{asset_type}**")

    colA, colB = st.columns([1, 1])
    with colA:
        st.markdown("### Scores (0–100)")
        st.metric("Score global", f"{s['score_global']:.0f}/100")
        st.progress(int(clamp(s["score_global"])))
        subs = {k: round(v, 0) for k, v in s.items() if k != "score_global"}
        st.write(subs)

    with colB:
        st.markdown("### Principales métriques (si disponibles)")
        def pct(x):
            return None if x is None else f"{100*float(x):.2f}%"
        st.write({
            "Nom": m.get("long_name"),
            "Secteur": m.get("sector"),
            "Industrie": m.get("industry"),
            "Revenue": m.get("revenue"),
            "Résultat net": m.get("net_income"),
            "Free Cash-Flow": m.get("fcf"),
            "Marge brute": pct(m.get("gross_margin")),
            "Marge nette": pct(m.get("net_margin")),
            "Marge FCF": pct(m.get("fcf_margin")),
            "ROE": pct(m.get("roe")),
            "Dette / EBITDA": m.get("debt_to_ebitda"),
            "PER": m.get("pe"),
            "P/FCF": m.get("p_fcf"),
            "CA CAGR ~2 ans": pct(m.get("rev_cagr")),
            "FCF CAGR ~2 ans": pct(m.get("fcf_cagr")),
        })

    st.markdown("### Commentaire")
    if asset_type == "ACTION":
        st.write(rule_based_comment(s, m))
    else:
        st.write("Analyse limitée.")

    if ai_provider == "Ollama (gratuit)":
        ai = ollama_investment_report(asset_type, ticker, isin, s, m, extra)
        if ai:
            st.info(ai)
        else:
            st.warning("Ollama ne répond pas. Vérifie qu’Ollama est lancé (localhost:11434).")

    elif ai_provider == "OpenAI":
        ai = openai_investment_report(asset_type, ticker, isin, s, m, extra)
        if ai:
            st.info(ai)
        else:
            st.warning("OPENAI_API_KEY non configurée ou appel OpenAI en échec.")

# ---- Settings tab
with tabs[2]:
    st.subheader("Paramètres & configuration")
    st.markdown("""
**Lancer**
- `streamlit run app.py`

**Ollama**
- Démarre Ollama: `ollama serve`
- Assure-toi qu’un modèle est dispo: `ollama pull mistral`

**OpenAI (optionnel)**
- `export OPENAI_API_KEY="sk-..."`
- `export OPENAI_MODEL="gpt-4.1-mini"`
""")