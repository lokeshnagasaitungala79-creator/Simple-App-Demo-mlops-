import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:1234")

client = MlflowClient()

models = client.search_model_versions(
    "name='ElasticNetWineModel'"
)

for model in models:
    print(
        "Version:",
        model.version,
        "Run ID:",
        model.run_id,
        "Stage:",
        model.current_stage
    )