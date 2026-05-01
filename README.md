# temporal

Predict next-day top movers among the largest cryptocurrencies.

`predict_movers.py` pulls daily prices and volumes from the public
CoinGecko API for the top-N coins by market cap, computes a small set
of momentum and volatility features (1/3/7-day returns, realized
volatility, RSI-14, short/long volume ratio), and ranks coins by
expected absolute next-day move. Each row is also tagged with a
directional `LEAN` (up / down / flat) blending momentum and RSI mean
reversion.

## Usage

```
python3 predict_movers.py             # top 10 from the top-50 universe, 30d lookback
python3 predict_movers.py --top 15 --universe 100 --days 45
python3 predict_movers.py --json      # machine-readable output
```

No third-party dependencies (Python 3.10+ standard library only).

CoinGecko's free tier is rate-limited; the script paces requests with
`--sleep` (default 1.5s). A run over the top 50 coins takes ~80s.

## Method

For each coin we estimate the next-day expected absolute return as

```
E[|Δ|] = realized_vol_14 * (1 + max(volume_change, 0) * 0.5)
                          * (1 + |RSI - 50| / 50 * 0.4)
```

and a directional score as

```
direction = (0.5*r1d + 0.3*r3d + 0.2*r7d) / realized_vol  +  rsi_reversion
```

where `rsi_reversion` pulls the lean opposite of an extreme RSI
(>=75 or <=25). Coins are sorted by `E[|Δ|]` descending — those are
the predicted top movers.

This is a transparent heuristic, not a trained model. It is intended
as a research / exploration tool. **Not financial advice.**
