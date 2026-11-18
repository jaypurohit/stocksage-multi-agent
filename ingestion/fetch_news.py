import os
import json
import time
import requests
from dotenv import load_dotenv
from config import ALLOWED_SYMBOLS, NEWS_BRONZE_PATH, NEWS_FROM, NEWS_TO

load_dotenv()  # loads FINNHUB_API_KEY from .env


class FetchNews:
    """
    Bronze ingestion for the news track: pull company news per ticker from
    Finnhub and write it raw to JSON, one file per ticker. Same bronze
    principle as prices — capture exactly what the API returns, immutable,
    no cleaning here.
    """

    BASE_URL = "https://finnhub.io/api/v1/company-news"

    def __init__(self, symbols, from_date, to_date, save_path):
        self.symbols = symbols
        self.from_date = from_date
        self.to_date = to_date
        self.save_path = save_path
        self.api_key = os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            raise SystemExit("Set FINNHUB_API_KEY in your .env file.")

    def _fetch_one(self, symbol):
        """Call Finnhub's company-news endpoint for one ticker."""
        params = {
            "symbol": symbol,
            "from": self.from_date,
            "to": self.to_date,
            "token": self.api_key,
        }
        # requests builds the URL + query string and makes the HTTP GET
        response = requests.get(self.BASE_URL, params=params, timeout=30)
        response.raise_for_status()      # raise if the HTTP call failed (4xx/5xx)
        return response.json()           # Finnhub returns a JSON array of articles

    def download_news(self):
        os.makedirs(self.save_path, exist_ok=True)
        for symbol in self.symbols:
            print(f"Fetching news for {symbol}...")
            try:
                articles = self._fetch_one(symbol)
            except Exception as e:
                print(f"  failed for {symbol}: {e}")
                continue

            if articles:
                out_path = os.path.join(self.save_path, f"{symbol}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(articles, f, ensure_ascii=False, indent=2)
                print(f"  saved {len(articles)} articles → {out_path}")
            else:
                print(f"  no articles returned for {symbol}")

            time.sleep(1.1)   # stay under Finnhub's free-tier rate limit (60/min)


if __name__ == "__main__":
    fetcher = FetchNews(
        symbols=ALLOWED_SYMBOLS,
        from_date=NEWS_FROM,
        to_date=NEWS_TO,
        save_path=NEWS_BRONZE_PATH,
    )
    fetcher.download_news()