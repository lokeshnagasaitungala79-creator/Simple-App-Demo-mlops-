import os
from flask import Flask, jsonify, render_template, request
from prediction_service import prediction

webapp_root = "webapp"
static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(
    __name__, static_folder=static_dir, template_folder=template_dir
)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            print("FORM DATA:", dict(request.form))

            data_req = dict(request.form)
            response = prediction.form_response(data_req)

            print("PREDICTION:", response)

            return render_template("index.html", response=response)

        except Exception as e:
            print("ERROR TYPE:", type(e).__name__)
            print("ERROR MESSAGE:", repr(e))

            return f"""
            <h1>Prediction Error</h1>
            <p><b>Error Type:</b> {type(e).__name__}</p>
            <p><b>Error:</b> {e}</p>
            """, 400

    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()
        response = prediction.api_response(data)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)