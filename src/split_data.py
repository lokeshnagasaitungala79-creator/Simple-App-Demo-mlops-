#split the raw data into train and test data
#save it
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from get_data import get_data

def split_and_save_data(config_path):
    config = get_data(config_path)
    test_data_path = config["split_data"]["test_data_path"]
    train_data_path = config["split_data"]["train_data_path"]
    raw_data_path = config["split_data"]["raw_data_csv"]
    split_ratio = config["split_data"]["test_size"]
    random_state = config["split_data"]["random_state"]

    df = pd.read_csv(raw_data_path , sep=',')
    train,test = train_test_split(df , test_size=split_ratio , random_state = random_state)
    train.to_csv(train_data_path, index=False , sep=',')
    test.to_csv(test_data_path, index=False, sep=',')

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config")
    parsed_args = args.parse_args()
    split_and_save_data(config_path=parsed_args.config)