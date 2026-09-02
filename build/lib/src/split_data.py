# split the raw data into train and test data
# save it

import os
import argparse
from get_data import read_params, get_data
from sklearn.model_selection import train_test_split


def split_and_save_data(config_path):
    # Read YAML configuration
    config = read_params(config_path)

    # Get paths from params.yaml
    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]

    test_size = config["split_data"]["test_size"]

    # Load the raw dataset
    df = get_data(config_path)

    # Split data
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=42
    )

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(train_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_data_path), exist_ok=True)

    # Save train and test data
    train.to_csv(train_data_path, index=False)
    test.to_csv(test_data_path, index=False)

    print("Data splitting completed successfully!")
    print("Train data shape:", train.shape)
    print("Test data shape:", test.shape)


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument(
        "--config",
        default="params.yaml"
    )

    parsed_args = args.parse_args()

    split_and_save_data(
        config_path=parsed_args.config
    )