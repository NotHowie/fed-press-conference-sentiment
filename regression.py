import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm

sent = pd.read_csv("fomc_meeting_sentiment_summary.csv")
sent["meeting_date"] = pd.to_datetime(sent["meeting_mmddyy"], format="mixed")

etfs = ["GOVT", "SGOV", "IEF", "TLT"]

raw = yf.download(
    etfs,
    start="2023-01-01",
    end="2025-12-31",
    auto_adjust=True,
    group_by="column"
)

prices = raw["Close"].copy().reset_index().rename(columns={"Date": "date"})
prices = prices.sort_values("date").reset_index(drop=True)

k = 7
for t in etfs:
    prices[f"{t}_ret_{k}d"] = prices[t].shift(-k) / prices[t] - 1

sent = sent.sort_values("meeting_date").reset_index(drop=True)
sent["event_date"] = sent["meeting_date"].apply(
    lambda d: prices.loc[prices["date"] > d, "date"].iloc[0]
)

ret_cols = ["date"] + [f"{t}_ret_{k}d" for t in etfs]
df = sent.merge(prices[ret_cols], left_on="event_date", right_on="date", how="left")

specs = {
    "caution_ratio": ["caution_ratio"],
    "weighted_tone": ["weighted_tone"],
    "tails_only": ["neg_tail_share", "pos_tail_share"],
    "tone_and_tails": ["weighted_tone", "neg_tail_share", "pos_tail_share"],
}

all_results = []

for spec_name, x_vars in specs.items():
    for t in etfs:
        y_col = f"{t}_ret_{k}d"
        sub = df.dropna(subset=[y_col] + x_vars).copy()

        if len(sub) < (len(x_vars) + 5):
            all_results.append({"spec": spec_name, "ETF": t, "n": int(len(sub)), "r2": np.nan})
            continue

        X = sm.add_constant(sub[x_vars])
        y = sub[y_col]
        m = sm.OLS(y, X).fit()

        row = {"spec": spec_name, "ETF": t, "n": int(m.nobs), "r2": float(m.rsquared)}
        for v in x_vars:
            row[f"beta_{v}"] = float(m.params[v])
            row[f"p_{v}"] = float(m.pvalues[v])

        all_results.append(row)

out = pd.DataFrame(all_results)

out.to_csv("sentiment_etf_event_regression_results_post_event.csv", index=False)
print("\nSaved: sentiment_etf_event_regression_results_post_event.csv")
