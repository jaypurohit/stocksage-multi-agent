import os
import pandas as pd
from config import ALLOWED_SYMBOLS, BRONZE_PATH, SILVER_PATH, START_DATE, END_DATE


class PricesSilver:
    """
    Silver layer: read raw bronze price Parquet, clean + conform it, run a
    validation gate (logging rejects rather than silently dropping), and write
    typed, deduplicated Parquet per ticker.

    Reads BRONZE only — never the API. If this logic is wrong, we re-run against
    immutable bronze; we never re-fetch. That separation is the whole point.
    """

    # the clean schema every silver file conforms to
    EXPECTED_COLS = ["date", "open", "high", "low", "close", "volume"]

    def __init__(self, symbols, bronze_path, silver_path, start_date, end_date):
        self.symbols = symbols
        self.bronze_path = bronze_path
        self.silver_path = silver_path
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

    # ---------- the per-ticker transform ----------

    def _load_and_flatten(self, symbol):
        """Read one bronze file and normalize its shape to the clean schema."""
        path = os.path.join(self.bronze_path, f"{symbol}.parquet")
        df = pd.read_parquet(path)

        # yfinance stores Date as the INDEX → make it a real column
        df = df.reset_index()

        # yfinance returns MultiIndex columns like ('Close','AAPL'); flatten to
        # the field name only, lowercased → 'close'. 'Date' → 'date'.
        df.columns = [
            c[0].lower() if isinstance(c, tuple) else str(c).lower()
            for c in df.columns
        ]
        return df

    def _conform_types(self, df):
        """Enforce types so downstream (gold) can trust the schema."""
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()  # day granularity
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype("float64")
        df["volume"] = df["volume"].astype("int64")
        return df

    # ---------- the validation gate ----------

    def _validate(self, df, symbol):
        """
        Split df into valid rows and rejects. We LOG rejects (count + reason)
        rather than silently dropping them — silent data loss is the thing a
        data pipeline must never do.
        Returns (clean_df, reject_count).
        """
        # build a boolean mask of rows that pass EVERY rule
        not_null   = df["date"].notna() & df["close"].notna()
        ohlc_high  = (df["high"] >= df["low"]) & (df["high"] >= df["open"]) & (df["high"] >= df["close"])
        ohlc_low   = (df["low"] <= df["open"]) & (df["low"] <= df["close"])
        vol_ok     = df["volume"] >= 0
        in_window  = (df["date"] >= self.start_date) & (df["date"] <= self.end_date)

        valid_mask = not_null & ohlc_high & ohlc_low & vol_ok & in_window
        rejects = df[~valid_mask]

        # report WHY rows were rejected (helps debugging, shows data quality)
        if len(rejects) > 0:
            print(f"    ⚠️  {symbol}: {len(rejects)} rows rejected")
            print(f"        null date/close : {(~not_null).sum()}")
            print(f"        OHLC sanity     : {(~(ohlc_high & ohlc_low)).sum()}")
            print(f"        bad volume      : {(~vol_ok).sum()}")
            print(f"        out of window   : {(~in_window).sum()}")

        return df[valid_mask], len(rejects)

    # ---------- orchestration ----------

    def process_symbol(self, symbol):
        df = self._load_and_flatten(symbol)
        df["ticker"] = symbol                       # self-describing: symbol in the data
        df = self._conform_types(df)

        # dedupe on the natural key (one row per ticker per day)
        before = len(df)
        df = df.drop_duplicates(subset=["ticker", "date"])
        dupes = before - len(df)

        clean_df, rejected = self._validate(df, symbol)

        # column order: keys first, then OHLCV
        clean_df = clean_df[["ticker"] + self.EXPECTED_COLS]

        out_path = os.path.join(self.silver_path, f"{symbol}.parquet")
        clean_df.to_parquet(out_path, index=False)

        print(f"  {symbol}: in={before}  dupes={dupes}  rejected={rejected}  out={len(clean_df)}")
        return len(clean_df)

    def run(self):
        os.makedirs(self.silver_path, exist_ok=True)
        print("Building silver layer...")
        total = 0
        for symbol in self.symbols:
            total += self.process_symbol(symbol)
        print(f"\nDone. {total} clean rows across {len(self.symbols)} tickers → {self.silver_path}")


if __name__ == "__main__":
    silver = PricesSilver(
        symbols=ALLOWED_SYMBOLS,
        bronze_path=BRONZE_PATH,
        silver_path=SILVER_PATH,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    silver.run()