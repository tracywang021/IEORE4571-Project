# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:54:24 2022

@author: Tracy
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description='a script to process raw data')
    parser.add_argument('--read_path', help='path to read raw data')
    args = parser.parse_args()
    return args

def extract_output_paths(read_path: str):
    """
    return output paths for train and test data
    args:
        - read_path: the path to read raw data "data/raw/xxx.csv"
    return:
        - train_output_path: path to save processed train data
        - test_output_path: path to save processed test data
    """
    level1_dir, _, file = read_path.split('/')
    level2_dir = 'processed'
    file_name, ext = file.split('.')
    train_output_path = level1_dir + '/' + level2_dir + '/' + file_name + '_train.' + ext
    test_output_path = level1_dir + '/' + level2_dir + '/' + file_name + '_test.' + ext
    return train_output_path, test_output_path

def process_data(read_path: str):
    """
    proprocess raw market data from yahoo finance and
    split data into training and testing set
    args:
        - read_path: the path to read raw data
        - output_path: the path to output the proce
    """
    df = pd.read_csv(read_path)
    
    # rename cols 
    rename_dict = {"Date": "date", "Open": "open", "High": "high",
               "Low": "low", "Close": "close", "Volume": "volume"}
    df.rename(columns=rename_dict, inplace=True)
    
    # compute next day i+1 return, annualized 
    # we assume a strategy of buying at open and selling at close
    df['open_i+1'] = df['open'].shift(-1)
    df['close_i+1'] = df['close'].shift(-1)
    df.dropna(inplace=True)
    annualized_fctr = 252
    df['return_i+1'] = (df['close_i+1']-df['open_i+1'])/df['open_i+1']*annualized_fctr
    
    # add clustering attributes 
    # open, high, low, close, volume from the day before
    df['open_i-1'] = df['open'].shift(1)
    df['high_i-1'] = df['high'].shift(1)
    df['low_i-1'] = df['low'].shift(1)
    df['close_i-1'] = df['close'].shift(1)
    df['volume_i-1'] = df['volume'].shift(1)
    df.dropna(inplace=True)
    # ratio between today and yesterday
    df['open/open_i-1'] = df['open']/df['open_i-1']
    df['high/high_i-1'] = df['high']/df['high_i-1']
    df['low/low_i-1'] = df['low']/df['low_i-1']
    df['close/close_i-1'] = df['close']/df['close_i-1']
    df['volume/volume_i-1'] = df['volume']/df['volume_i-1']
    
    # add clustering attributes from day i-2
    df['open_i-2'] = df['open'].shift(2)
    df['high_i-2'] = df['high'].shift(2)
    df['low_i-2'] = df['low'].shift(2)
    df['close_i-2'] = df['close'].shift(2)
    df['volume_i-2'] = df['volume'].shift(2)
    df.dropna(inplace=True)
    # ratio between day i-1 and day i-2
    # ratio between day i-1 and i-2
    df['open_i-1/open_i-2'] = df['open_i-1']/df['open_i-2']
    df['high_i-1/high_i-2'] = df['high_i-1']/df['high_i-2']
    df['low_i-1/low_i-2'] = df['low_i-1']/df['low_i-2']
    df['close_i-1/close_i-2'] = df['close_i-1']/df['close_i-2']
    df['volume_i-1/volume_i-2'] = df['volume_i-1']/df['volume_i-2']
    
    # split data into train and test set
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    # save data to output_path
    train_output_path, test_output_path = extract_output_paths(read_path)
    train.to_csv(train_output_path)
    test.to_csv(test_output_path)
    return

if __name__ == '__main__':
    args = parse_args()
    process_data(args.read_path)