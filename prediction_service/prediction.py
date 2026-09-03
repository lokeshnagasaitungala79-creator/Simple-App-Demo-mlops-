import os
import json
import joblib
import numpy as np
import yaml

params_path = "params.yaml"
schema_path = os.path.join("prediction_service", "schema_in.json")


class NotInRange(Exception):
    def __init__(self, message="Values entered are not in range"):
        self.message = message
        super().__init__(self.message)


class NotInCols(Exception):
    def __init__(self, message="Column not found in schema"):
        self.message = message
        super().__init__(self.message)


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def predict(data):
    config = read_params(params_path)
    model_dir_path = config["webapp_model_dir"]
    model = joblib.load(model_dir_path)

    prediction = model.predict(data).tolist()[0]

    if 3 <= prediction <= 8:
        return prediction

    raise NotInRange("Unexpected prediction result")


def get_schema(schema_path=schema_path):
    with open(schema_path) as json_file:
        schema = json.load(json_file)
    return schema


def validate_input(dict_request):
    schema = get_schema()
    columns = schema["columns"]

    for col, val in dict_request.items():
        if col not in columns:
            raise NotInCols(f"{col} is not a valid column")

        value = float(val)

        if value < columns[col]["min"] or value > columns[col]["max"]:
            raise NotInRange(
                f"{col} should be between "
                f"{columns[col]['min']} and {columns[col]['max']}"
            )

    return True


def form_response(dict_request):
    print("FORM DATA:", dict_request)

    if validate_input(dict_request):
        data = [list(map(float, dict_request.values()))]

        print("MODEL INPUT:", data)

        response = predict(data)

        print("PREDICTION:", response)

        return response


def api_response(dict_request):
    try:
        if validate_input(dict_request):
            data = np.array(
                [list(map(float, dict_request.values()))]
            )
            response = predict(data)
            return {"response": response}

    except Exception as e:
        return {
            "expected_range": get_schema(),
            "response": str(e)
        }