from flask import Flask, render_template, request, jsonify
import os
import yaml
import joblib
import numpy as np

params_path = "params.yaml"
webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def predict(data):
    config = read_params(params_path)
    model_dir_path = config["webapp_model_dir"]
    model = joblib.load(model_dir_path)
    prediction = model.predict(data)
    return prediction[0]

def api_response(req):
    try:
        data = np.array([list(req.json.values())], dtype=float)
        response = predict(data)
        return {"response": response}
    except Exception as e:
        print("Error:", e)
        return {"error": "Something went wrong! Try again later."}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            if request.is_json:
                response = api_response(request)
                return jsonify(response)
            elif request.form:
                data = list(request.form.values())
                data = [list(map(float, data))]
                response = predict(data)
                return render_template("index.html", response=response)
        except Exception as e:
            print("Error:", e)
            error = {"error": "Something went wrong! Try again later."}
            return jsonify(error), 400
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)