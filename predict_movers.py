"""Predict next-day top movers among major cryptocurrencies.

Pulls recent OHLC/volume data from the public CoinGecko API, computes a
small set of momentum and volatility features, and ranks coins by an
expected absolute next-day move. The output highlights both the most
likely large movers and the directional lean (up vs. down).

This is a transparent heuristic, not a trained model. Do not use the
output as financial advice.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
USER_AGENT = "crypto-top-movers/0.1 (+https://github.com/roman-bm/temporal)"


@dataclasses.dataclass
class CoinFeatures:
    coin_id: str
    symbol: str
    name: str
    price: float
    ret_1d: float
    ret_3d: float
    ret_7d: float
    realized_vol: float
    vol_change: float
    rsi_14: float
    expected_abs_move: float
    directional_score: float

    @property
    def lean(self) -> str:
        if self.directional_score > 0.15:
            return "up"
        if self.directional_score < -0.15:
            return "down"
        return "flat"


def http_get_json(url: str, retries: int = 4, backoff: float = 1.5) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_exc = exc
            sleep = backoff ** attempt
            time.sleep(sleep)
    assert last_exc is not None
    raise last_exc


def fetch_top_coins(limit: int, vs_currency: str = "usd") -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "vs_currency": vs_currency,
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "24h",
        }
    )
    return http_get_json(f"{COINGECKO_BASE}/coins/markets?{params}")


def fetch_market_chart(coin_id: str, days: int, vs_currency: str = "usd") -> dict[str, Any]:
    params = urllib.parse.urlencode({"vs_currency": vs_currency, "days": days, "interval": "daily"})
    return http_get_json(f"{COINGECKO_BASE}/coins/{coin_id}/market_chart?{params}")


def daily_series(chart: dict[str, Any]) -> tuple[list[float], list[float]]:
    prices = [p[1] for p in chart.get("prices", []) if p and p[1] is not None]
    volumes = [v[1] for v in chart.get("total_volumes", []) if v and v[1] is not None]
    n = min(len(prices), len(volumes))
    return prices[-n:], volumes[-n:]


def pct_return(series: list[float], lookback: int) -> float:
    if len(series) <= lookback or series[-1 - lookback] == 0:
        return 0.0
    return (series[-1] / series[-1 - lookback]) - 1.0


def realized_vol(prices: list[float], window: int = 14) -> float:
    if len(prices) < window + 1:
        return 0.0
    rets = [
        math.log(prices[i] / prices[i - 1])
        for i in range(len(prices) - window, len(prices))
        if prices[i - 1] > 0
    ]
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var)


def rsi(prices: list[float], period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    gains: list[float] = []
    losses: list[float] = []
    for i in range(len(prices) - period, len(prices)):
        change = prices[i] - prices[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def volume_change(volumes: list[float], short: int = 3, long: int = 14) -> float:
    if len(volumes) < long:
        return 0.0
    short_avg = sum(volumes[-short:]) / short
    long_avg = sum(volumes[-long:]) / long
    if long_avg == 0:
        return 0.0
    return (short_avg / long_avg) - 1.0


def compute_features(meta: dict[str, Any], chart: dict[str, Any]) -> CoinFeatures | None:
    prices, volumes = daily_series(chart)
    if len(prices) < 15:
        return None

    r1 = pct_return(prices, 1)
    r3 = pct_return(prices, 3)
    r7 = pct_return(prices, 7)
    vol = realized_vol(prices, window=14)
    vchg = volume_change(volumes)
    r = rsi(prices)

    # Expected absolute move: the daily realized volatility is a baseline
    # 1-sigma estimate. Surging volume and stretched RSI tend to precede
    # larger-than-average moves, so we scale it up under those conditions.
    vol_boost = 1.0 + max(vchg, 0.0) * 0.5
    rsi_stretch = abs(r - 50.0) / 50.0
    stretch_boost = 1.0 + rsi_stretch * 0.4
    expected_abs = vol * vol_boost * stretch_boost

    # Directional lean: blend short-horizon momentum with mean-reversion
    # when RSI is at an extreme.
    momentum = 0.5 * r1 + 0.3 * r3 + 0.2 * r7
    if r >= 75:
        reversion = -0.5 * (r - 75) / 25
    elif r <= 25:
        reversion = 0.5 * (25 - r) / 25
    else:
        reversion = 0.0
    direction = momentum / max(vol, 1e-6) + reversion

    return CoinFeatures(
        coin_id=meta["id"],
        symbol=str(meta.get("symbol", "")).upper(),
        name=meta.get("name", meta["id"]),
        price=float(prices[-1]),
        ret_1d=r1,
        ret_3d=r3,
        ret_7d=r7,
        realized_vol=vol,
        vol_change=vchg,
        rsi_14=r,
        expected_abs_move=expected_abs,
        directional_score=direction,
    )


def predict(top_n: int, candidates: int, days: int, sleep: float) -> list[CoinFeatures]:
    coins = fetch_top_coins(candidates)
    out: list[CoinFeatures] = []
    for i, meta in enumerate(coins, 1):
        try:
            chart = fetch_market_chart(meta["id"], days=days)
        except urllib.error.HTTPError as exc:
            print(f"  ! {meta['id']}: HTTP {exc.code}", file=sys.stderr)
            time.sleep(sleep * 2)
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {meta['id']}: {exc}", file=sys.stderr)
            continue
        feats = compute_features(meta, chart)
        if feats is not None:
            out.append(feats)
        print(f"  [{i}/{len(coins)}] {meta['symbol'].upper():<6}  ok", file=sys.stderr)
        time.sleep(sleep)
    out.sort(key=lambda f: f.expected_abs_move, reverse=True)
    return out[:top_n]


def format_pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def render_table(rows: list[CoinFeatures]) -> str:
    headers = ["#", "SYM", "NAME", "PRICE", "1D", "3D", "7D", "VOL14", "VOLΔ", "RSI", "E[|Δ|]", "LEAN"]
    data = []
    for i, f in enumerate(rows, 1):
        data.append(
            [
                str(i),
                f.symbol,
                f.name[:18],
                f"${f.price:,.4f}" if f.price < 1 else f"${f.price:,.2f}",
                format_pct(f.ret_1d),
                format_pct(f.ret_3d),
                format_pct(f.ret_7d),
                f"{f.realized_vol * 100:.2f}%",
                format_pct(f.vol_change),
                f"{f.rsi_14:5.1f}",
                f"{f.expected_abs_move * 100:.2f}%",
                f.lean,
            ]
        )
    widths = [max(len(h), *(len(r[i]) for r in data)) if data else len(h) for i, h in enumerate(headers)]
    line = lambda cells: "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))
    sep = "  ".join("-" * w for w in widths)
    return "\n".join([line(headers), sep, *(line(r) for r in data)])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=10, help="number of predicted top movers to return")
    parser.add_argument(
        "--universe",
        type=int,
        default=50,
        help="how many top-market-cap coins to evaluate as candidates",
    )
    parser.add_argument("--days", type=int, default=30, help="lookback window in days")
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.5,
        help="seconds between API calls (CoinGecko free tier is rate-limited)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args(argv)

    print(
        f"Fetching top {args.universe} coins, {args.days}d window...",
        file=sys.stderr,
    )
    rows = predict(top_n=args.top, candidates=args.universe, days=args.days, sleep=args.sleep)

    if args.json:
        print(json.dumps([dataclasses.asdict(r) | {"lean": r.lean} for r in rows], indent=2))
    else:
        print()
        print(f"Predicted top {len(rows)} movers for the next 24h")
        print("(ranked by expected absolute move; LEAN is directional bias)")
        print()
        print(render_table(rows))
        print()
        print("Heuristic only. Not financial advice.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
