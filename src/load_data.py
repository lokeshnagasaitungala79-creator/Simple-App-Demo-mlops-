#read the data from the data source
#save it in the data/raw for the further process
import os
from get_data import read_params , get_data
import argparse

def load_and_save_data(config_path):
    config = read_params(config_path)
    df = get_data(config_path)
    new_columns = [col.replace(" ", "_") for col in df.columns]
    raw_data_path = config["data_source"]["raw_dataset_csv"]
    df.to_csv(raw_data_path, header=True, index=False, sep=",")
    df.head()
    print(new_columns)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config" , default = "params.yaml")
    parsed_args = args.parse_args()
    data = load_and_save_data(config_path = parsed_args.config)
