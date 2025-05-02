import yfinance as yf

def get_stock_data(ticker="AAPL", start="2015-01-01", end="2024-12-31"):
    df = yf.download(ticker, start=start, end=end)
    return df[['Close']]
