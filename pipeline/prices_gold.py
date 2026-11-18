import os
import duckdb
from config import SILVER_PATH, GOLD_PATH


class PricesGold:
    """
    Gold layer: read ALL silver files at once, compute analytics with SQL
    window functions, write one curated, analytics-ready table.

    Reads SILVER only — never bronze, never the API. Gold is derived; if the
    analytics logic changes, we re-run gold against silver without touching
    the raw or cleaned layers. That's the medallion separation.

    DuckDB does the computation: it reads Parquet directly and runs window
    functions far more cleanly than row-by-row Python.
    """

    def __init__(self, silver_path, gold_path):
        self.silver_path = silver_path
        self.gold_path = gold_path
        # glob pattern: read every ticker's silver Parquet in one shot
        self.silver_glob = os.path.join(silver_path, "*.parquet")
        self.out_path = os.path.join(gold_path, "prices_daily.parquet")

    # The analytics query — the heart of gold.
    # Window functions compute across rows WITHOUT collapsing them, so every
    # (ticker, date) row survives, each enriched with its own metrics.
    def _query(self):
        return f"""
        WITH with_returns AS (
            SELECT
                ticker, date, open, high, low, close, volume,

                -- previous day's close (1 row back, within this ticker, by date)
                LAG(close) OVER w AS prev_close,

                -- daily return = today / yesterday - 1  (NULL on each ticker's first day)
                close / LAG(close) OVER w - 1 AS daily_return,

                -- 7-day moving average: this row + 6 before it (a SLIDING window).
                -- ROWS BETWEEN ... is what makes it 'moving' instead of a running total.
                AVG(close) OVER (
                    PARTITION BY ticker ORDER BY date
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) AS ma_7,

                -- 30-day moving average: this row + 29 before it
                AVG(close) OVER (
                    PARTITION BY ticker ORDER BY date
                    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                ) AS ma_30,

                -- cumulative return from each ticker's first day (running, default frame)
                close / FIRST_VALUE(close) OVER w - 1 AS cumulative_return

            FROM read_parquet('{self.silver_glob}')
            -- named window, defined once and reused as 'OVER w'
            WINDOW w AS (PARTITION BY ticker ORDER BY date)
        )
        SELECT
            *,
            -- 30-day rolling volatility = std dev of daily_return over the window.
            -- Computed in the OUTER query because you can't nest window functions:
            -- daily_return is itself a window result, so we layer with a CTE.
            STDDEV(daily_return) OVER (
                PARTITION BY ticker ORDER BY date
                ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
            ) AS volatility_30
        FROM with_returns
        ORDER BY ticker, date
        """

    def run(self):
        os.makedirs(self.gold_path, exist_ok=True)
        con = duckdb.connect()   # in-memory DuckDB; reads Parquet directly

        print("Building gold layer...")

        # COPY (query) TO ... writes the query result straight to Parquet.
        con.execute(
            f"COPY ({self._query()}) TO '{self.out_path}' (FORMAT PARQUET)"
        )

        # --- verification: read back a summary so you can confirm it worked ---
        total = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{self.out_path}')"
        ).fetchone()[0]

        print(f"Done. {total} rows → {self.out_path}\n")

        # peek at NVDA's most recent 5 days to eyeball the analytics
        print("Sample (NVDA, latest 5 days):")
        sample = con.execute(f"""
            SELECT date, close,
                   ROUND(daily_return, 4)   AS daily_return,
                   ROUND(ma_7, 2)           AS ma_7,
                   ROUND(ma_30, 2)          AS ma_30,
                   ROUND(volatility_30, 4)  AS volatility_30,
                   ROUND(cumulative_return, 4) AS cum_return
            FROM read_parquet('{self.out_path}')
            WHERE ticker = 'NVDA'
            ORDER BY date DESC
            LIMIT 5
        """).fetchdf()
        print(sample.to_string(index=False))

        con.close()


if __name__ == "__main__":
    gold = PricesGold(silver_path=SILVER_PATH, gold_path=GOLD_PATH)
    gold.run()