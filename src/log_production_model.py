import argparse
import os
import joblib
import mlflow
from mlflow.tracking import MlflowClient
from src.get_data import read_params


def log_production_model(config_path):
    config = read_params(config_path)

    mlflow_config = config["mlflow_config"]
    # Check key name matching params.yaml (registered_model vs registered_model_name)
    model_name = mlflow_config.get("registered_model") or mlflow_config.get("registered_model_name")
    remote_server_url = mlflow_config["remote_server_url"]

    mlflow.set_tracking_uri(remote_server_url)
    mlflow.set_registry_uri(remote_server_url)

    # Get experiment ID by name instead of hardcoding experiment_ids=1
    experiment = mlflow.get_experiment_by_name(mlflow_config["experiment_name"])
    if experiment is None:
        raise ValueError(f"Experiment '{mlflow_config['experiment_name']}' not found.")
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    # Find the run with the lowest MAE
    lowest_mae_run = runs.sort_values(by="metrics.mae", ascending=True).iloc[0]
    lowest_run_id = lowest_mae_run["run_id"]

    client = MlflowClient()

    for mv in client.search_model_versions(f"name='{model_name}'"):
        mv_dict = dict(mv)

        if mv_dict["run_id"] == lowest_run_id:
            current_version = mv_dict["version"]
            logged_model = mv_dict["source"]
            
            # Transition best model to Production
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Production",
                archive_existing_versions=True
            )
        else:
            current_version = mv_dict["version"]
            # Transition other versions to Staging
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Staging"
            )

    # Load production model using the best run's artifact patha
    model_uri = f"runs:/{lowest_run_id}/model"
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    webapp_model_path = config["webapp_model_dir"]
    os.makedirs(os.path.dirname(webapp_model_path), exist_ok=True)

    joblib.dump(loaded_model, webapp_model_path)
    print(f"Successfully updated production model at: {webapp_model_path}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    log_production_model(config_path=parsed_args.config)