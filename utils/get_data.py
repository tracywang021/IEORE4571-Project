# -*- coding: utf-8 -*-
"""
Get historical market data given a ticker
"""
import argparse
import yfinance as yf

def parse_args():
    parser = argparse.ArgumentParser(description="a script to get historical market data")
    parser.add_argument("--ticker", help='ticker symbol')
    parser.add_argument("--output_path", help='output path of the historical data')
    args = parser.parse_args()
    return args
    

def get_data(ticker: str, output_path:str, period="max"):
    """
    get and save historical market data for a specific ticker
    args:
        - ticker: ticker symbol of a particular stock
        - output_path: the path to save the extracted data under main dir
        - period: time span of market data. default: max. 
            other options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, and ytd
    """
    # get historical data 
    stock = yf.Ticker(ticker)
    hist_df = stock.history(period=period)
    
    # keep Open, High, Low, Close, Volume cols
    use_cols = ["Open", "High", "Low", "Close", "Volume"]
    hist_df = hist_df[use_cols]
    
    # save data to output_path
    hist_df.to_csv(output_path)
    return 

if __name__ == '__main__':
    args = parse_args()
    get_data(args.ticker, args.output_path)

