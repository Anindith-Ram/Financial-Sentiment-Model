from typing import List


def get_sp500_tickers() -> List[str]:
    """Return the current S&P 500 constituents without HTML scraping.

    Uses yfinance's hard-coded static list (updated regularly in the package).
    No network call → no API key required.
    """
    try:
        import yfinance as yf  # deferred import
        if hasattr(yf, "tickers_sp500"):
            return yf.tickers_sp500()
    except ImportError:
        # We'll fall back to web scrape below
        pass

    # Fallback: scrape Wikipedia table
    import pandas as pd
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        table = pd.read_html(url, header=0)[0]
        symbols = table["Symbol"].tolist()
        # Clean up symbols with dots (e.g., BRK.B -> BRK-B for yfinance)
        symbols = [s.replace(".", "-") for s in symbols]
        return symbols
    except Exception as e:
        raise RuntimeError("Failed to fetch S&P 500 tickers via yfinance and Wikipedia scrape") from e

if __name__ == '__main__':
    tickers = get_sp500_tickers()
    print(tickers) 