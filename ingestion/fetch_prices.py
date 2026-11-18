import os
import yfinance as yf
from config import ALLOWED_SYMBOLS, BRONZE_PATH, START_DATE,END_DATE
os.makedirs(BRONZE_PATH, exist_ok=True)

class FetchPrices:

    def __init__(self, symbols, start_date, end_date, save_path):
        """Initializes the fetcher with necessary configurations."""
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.save_path = save_path

    def download_prices(self):
        os.makedirs(self.save_path, exist_ok=True)
        for symbol in self.symbols:  # self, not the global
            print(f"Downloading data for {symbol}...")
            response = yf.download(
                tickers=symbol,
                start=self.start_date,
                end=self.end_date
            )
            if not response.empty:
                full_path = os.path.join(self.save_path, f"{symbol}.parquet")
                response.to_parquet(full_path)  # Parquet, not CSV
                print(f"Saved {symbol}: {len(response)} rows → {full_path}")
            else:
                print(f"No data for {symbol}, skipping.")


if __name__ == "__main__":
    fetcher = FetchPrices(
        symbols=ALLOWED_SYMBOLS,
        start_date=START_DATE,
        end_date=END_DATE,
        save_path=BRONZE_PATH
    )

    fetcher.download_prices()