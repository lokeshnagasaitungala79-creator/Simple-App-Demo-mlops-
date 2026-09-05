
import argparse
import json
import os
import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml

from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from urllib.parse import urlparse


def read_params(config_path):
    with open(config_path) as yaml_file:
        return yaml.safe_load(yaml_file)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)

    return rmse, mae, r2


def train_and_evaluate(config_path):
    config = read_params(config_path)

    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]

    alpha = config["estimators"]["ElasticNet"]["alpha"]
    l1_ratio = config["estimators"]["ElasticNet"]["l1_ratio"]

    model_dir = config["model_dir"]

    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    webapp_model_path = config["webapp_model_dir"]

    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)

    target = "quality"

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(columns=[target])
    test_x = test.drop(columns=[target])

    print("Training data shape:", train_x.shape)
    print("Testing data shape:", test_x.shape)

    # MLflow configuration
    mlflow_config = config["mlflow_config"]

    remote_server_url = mlflow_config["remote_server_url"]

    mlflow.set_tracking_uri(remote_server_url)

    mlflow.set_experiment(
        mlflow_config["experiment_name"]
    )

    with mlflow.start_run(
        run_name=mlflow_config["run_name"]
    ):

        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=42
        )

        model.fit(train_x, train_y)

        pred = model.predict(test_x)

        rmse, mae, r2 = eval_metrics(
            test_y,
            pred
        )

        print("ElasticNet Model")
        print("Alpha:", alpha)
        print("L1 Ratio:", l1_ratio)
        print("RMSE:", rmse)
        print("MAE:", mae)
        print("R2:", r2)

        # Log parameters
        mlflow.log_param(
            "alpha",
            alpha
        )

        mlflow.log_param(
            "l1_ratio",
            l1_ratio
        )

        # Log metrics
        mlflow.log_metric(
            "rmse",
            rmse
        )

        mlflow.log_metric(
            "mae",
            mae
        )

        mlflow.log_metric(
            "r2",
            r2
        )

        # Check MLflow artifact store
        tracking_url_type_store = urlparse(
            mlflow.get_artifact_uri()
        ).scheme

        if tracking_url_type_store != "file":

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=(
                    mlflow_config["registered_model_name"]
                )
            )

        else:

            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=(
                    mlflow_config["registered_model"]
                )
            )

        # Save metrics report
        scores_directory = os.path.dirname(
            scores_file
        )

        if scores_directory:
            os.makedirs(
                scores_directory,
                exist_ok=True
            )

        params_directory = os.path.dirname(
            params_file
        )

        if params_directory:
            os.makedirs(
                params_directory,
                exist_ok=True
            )

        with open(scores_file, "w") as f:
            json.dump(
                {
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2
                },
                f,
                indent=4
            )

        with open(params_file, "w") as f:
            json.dump(
                {
                    "alpha": alpha,
                    "l1_ratio": l1_ratio,
                    "random_state": 42
                },
                f,
                indent=4
            )

        # Save local model
        os.makedirs(
            model_dir,
            exist_ok=True
        )

        model_path = os.path.join(
            model_dir,
            "model.joblib"
        )

        joblib.dump(
            model,
            model_path
        )

        # Save model for web application
        webapp_directory = os.path.dirname(
            webapp_model_path
        )

        if webapp_directory:
            os.makedirs(
                webapp_directory,
                exist_ok=True
            )

        joblib.dump(
            model,
            webapp_model_path
        )

        print(
            "Model saved at:",
            model_path
        )

        print(
            "Webapp model saved at:",
            webapp_model_path
        )


if __name__ == "__main__":

    args = argparse.ArgumentParser()

    args.add_argument(
        "--config",
        default="params.yaml"
    )

    parsed_args = args.parse_args()

    train_and_evaluate(
        parsed_args.config
    )
