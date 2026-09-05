import argparse
import os
import joblib
import mlflow
from mlflow.tracking import MlflowClient

from src.get_data import read_params


def log_production_model(config_path):
    config = read_params(config_path)

    mlflow_config = config["mlflow_config"]

    model_name = (
        mlflow_config.get("registered_model")
        or mlflow_config.get("registered_model_name")
    )

    if not model_name:
        raise ValueError(
            "registered_model or registered_model_name is missing "
            "from params.yaml"
        )

    remote_server_url = mlflow_config["remote_server_url"]

    # Connect to MLflow
    mlflow.set_tracking_uri(remote_server_url)
    mlflow.set_registry_uri(remote_server_url)

    client = MlflowClient()

    # Get experiment
    experiment = mlflow.get_experiment_by_name(
        mlflow_config["experiment_name"]
    )

    if experiment is None:
        raise ValueError(
            f"Experiment '{mlflow_config['experiment_name']}' not found."
        )

    # Get all runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    if runs.empty:
        raise ValueError(
            "No MLflow runs found in the experiment."
        )

    # Check MAE column
    if "metrics.mae" not in runs.columns:
        raise ValueError(
            "No 'metrics.mae' metric found in MLflow runs."
        )

    # Find run with lowest MAE
    lowest_mae_run = runs.sort_values(
        by="metrics.mae",
        ascending=True
    ).iloc[0]

    lowest_run_id = lowest_mae_run["run_id"]

    print("Best run ID:", lowest_run_id)
    print("Best MAE:", lowest_mae_run["metrics.mae"])

    # ---------------------------------------------------------
    # STEP 1: Check whether the model is already registered
    # ---------------------------------------------------------

    model_versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    best_version = None

    for mv in model_versions:
        if mv.run_id == lowest_run_id:
            best_version = mv.version
            break

    # ---------------------------------------------------------
    # STEP 2: Register model if it is not already registered
    # ---------------------------------------------------------

    if best_version is None:

        print(
            "Best run is not registered. Registering model..."
        )

        model_source = f"runs:/{lowest_run_id}/model"

        try:
            registered_model = client.create_registered_model(
                model_name
            )

            print(
                f"Created registered model: {registered_model.name}"
            )

        except Exception:
            # Model already exists
            pass

        model_version = client.create_model_version(
            name=model_name,
            source=model_source,
            run_id=lowest_run_id
        )

        best_version = model_version.version

        print(
            f"Registered model version: {best_version}"
        )

    else:

        print(
            f"Model already registered. "
            f"Version: {best_version}"
        )

    # ---------------------------------------------------------
    # STEP 3: Move best model to Production
    # ---------------------------------------------------------

    print(
        f"Moving model version {best_version} "
        f"to Production..."
    )

    # MLflow 3.x uses aliases instead of stages.
    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=best_version
    )

    print(
        f"Model version {best_version} "
        f"assigned to 'production' alias."
    )

    # ---------------------------------------------------------
    # STEP 4: Load production model
    # ---------------------------------------------------------

    model_uri = (
        f"models:/{model_name}@production"
    )

    print(
        "Loading model from:",
        model_uri
    )

    loaded_model = mlflow.pyfunc.load_model(
        model_uri
    )

    # ---------------------------------------------------------
    # STEP 5: Save model for web application
    # ---------------------------------------------------------

    webapp_model_path = config["webapp_model_dir"]

    model_directory = os.path.dirname(
        webapp_model_path
    )

    if model_directory:
        os.makedirs(
            model_directory,
            exist_ok=True
        )

    joblib.dump(
        loaded_model,
        webapp_model_path
    )

    print(
        f"Successfully updated production model at: "
        f"{webapp_model_path}"
    )


if __name__ == "__main__":

    args = argparse.ArgumentParser()

    args.add_argument(
        "--config",
        default="params.yaml"
    )

    parsed_args = args.parse_args()

    log_production_model(
        config_path=parsed_args.config
    )
