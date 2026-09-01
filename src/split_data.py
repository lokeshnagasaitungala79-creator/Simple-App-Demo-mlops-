#split the raw data into train and test data
#save it
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from get_data import get_data

def split_and_save_data(config_path):
    config = get_data(config_path)
    test_data_path = config[""][""]
    train_data_path = config[""][""]
    raw_data_path = config[""][""]
    split_ratio = config[""][""]
    random_state = config[""][""]