from flask import Flask, render_template, request, jsonify
import os
from prediction_service import prediction

webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            if request.is_json:
                response = prediction.api_response(request.json)
                return jsonify(response)

            if request.form:
                data_req = dict(request.form)
                response = prediction.form_response(data_req)
                return render_template("index.html", response=response)

        except Exception as e:
            print("Error:", e)
            error = {"error": str(e)}
            return render_template("404.html", error=error), 400

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)