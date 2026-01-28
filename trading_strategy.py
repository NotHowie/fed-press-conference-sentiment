import pandas as pd
import numpy as np
import yfinance as yf

sent = pd.read_csv("fomc_meeting_sentiment_summary.csv")
sent["meeting_date"] = pd.to_datetime(sent["meeting_mmddyy"], format="mixed")

ticker = "SGOV"
raw = yf.download(
    [ticker],
    start="2023-01-01",
    end="2025-12-31",
    auto_adjust=True,
    group_by="column"
)

prices = raw["Close"].copy().reset_index().rename(columns={"Date": "date", ticker: "close"})
prices = prices.sort_values("date").reset_index(drop=True)

sent = sent.sort_values("meeting_date").reset_index(drop=True)
sent["event_date"] = sent["meeting_date"].apply(
    lambda d: prices.loc[prices["date"] > d, "date"].iloc[0]
)

k = 7
prices["ret_k"] = prices["close"].shift(-k) / prices["close"] - 1

df = sent.merge(prices[["date", "ret_k"]], left_on="event_date", right_on="date", how="left")

df = df.dropna(subset=["ret_k"]).copy()

med_tone = df["weighted_tone"].median()
med_pos_tail = df["pos_tail_share"].median()
med_caution = df["caution_ratio"].median()

df["pos_baseline"] = 1 
df["pos_tone"] = np.where(df["weighted_tone"] > med_tone, 1, 0)
df["pos_pos_tail"] = np.where(df["pos_tail_share"] > med_pos_tail, 1, 0)
df["pos_caution"] = np.where(df["caution_ratio"] > med_caution, -1, 0)

df["ret_baseline"] = df["pos_baseline"] * df["ret_k"]
df["ret_tone"] = df["pos_tone"] * df["ret_k"]
df["ret_pos_tail"] = df["pos_pos_tail"] * df["ret_k"]
df["ret_caution"] = df["pos_caution"] * df["ret_k"]

def summarize(name, r):
    n_trades = int((r != 0).sum())
    avg = float(r.mean())
    avg_trade = float(r[r != 0].mean()) if n_trades > 0 else np.nan
    hit = float((r[r != 0] > 0).mean()) if n_trades > 0 else np.nan
    cum = float((1 + r).prod() - 1)
    vol = float(r.std(ddof=1))
    sharpe = float(avg / vol) if vol and not np.isnan(vol) else np.nan

    return {
        "strategy": name,
        "k_days": k,
        "n_meetings": int(len(r)),
        "n_trades": n_trades,
        "avg_return_per_window": avg,
        "avg_return_per_trade": avg_trade,
        "hit_rate": hit,
        "cum_return": cum,
        "volatility": vol,
        "sharpe_like": sharpe,
    }

summary = pd.DataFrame([
    summarize("SGOV_baseline_long_all_meetings", df["ret_baseline"]),
    summarize("SGOV_long_when_weighted_tone_above_median", df["ret_tone"]),
    summarize("SGOV_long_when_pos_tail_above_median", df["ret_pos_tail"]),
    summarize("SGOV_short_when_caution_ratio_above_median", df["ret_caution"]),
])

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)
print("\n=== Strategy Performance (in-sample, post-conference, k=7) ===")
print(summary.to_string(index=False))