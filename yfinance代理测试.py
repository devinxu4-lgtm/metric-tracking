import yfinance as yf
import os
proxy = 'http://127.0.0.1:7890' # 代理设置，此处修改
os.environ['HTTP_PROXY'] = proxy
os.environ['HTTPS_PROXY'] = proxy
tickers = ['AAPL']
train_data = yf.download(tickers, start="2025-01-03", end="2025-07-01", interval="1d", auto_adjust=True)
print(train_data)
