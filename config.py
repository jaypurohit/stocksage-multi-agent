import os

# ---- Project root anchor ----
# __file__ is the path to THIS config.py. Since config.py lives at the project
# root, its directory IS the project root. Every path below is built from it as
# an ABSOLUTE path, so scripts resolve data the same way no matter where they
# run from — no more "current directory" surprises, no manual moving.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---- Tickers & date windows ----
ALLOWED_SYMBOLS = (
    "AAPL", "MSFT", "GOOG", "TSLA", "AMZN",
    "META", "NVDA", "TSM", "HPE", "MU",
)

START_DATE = "2024-09-03"     # prices: 2-year daily window
END_DATE = "2026-09-03"

NEWS_FROM = "2026-05-01"      # news: recent window (free-tier history limit)
NEWS_TO = "2026-11-01"

# ---- Paths (all absolute, anchored to PROJECT_ROOT) ----
BRONZE_PATH      = os.path.join(PROJECT_ROOT, "data", "bronze", "prices")
SILVER_PATH      = os.path.join(PROJECT_ROOT, "data", "silver", "prices")
GOLD_PATH        = os.path.join(PROJECT_ROOT, "data", "gold")
NEWS_BRONZE_PATH = os.path.join(PROJECT_ROOT, "data", "bronze", "news")
CHROMA_DIR       = os.path.join(PROJECT_ROOT, "data", "chroma_db")

EMBEDDING_MODEL = "text-embedding-3-small"
NEWS_COLLECTION = "financial_news"

MEDICAL_DATA = os.path.join(PROJECT_ROOT, "data", "medical", "abstracts.json")
MEDICAL_COLLECTION = "pubmed_abstracts"