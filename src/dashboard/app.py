#!/usr/bin/env python3
"""
src/dashboard/app.py

Flask dashboard that interfaces with the Diabetes Risk Prediction pipeline.

Run (development):
    1. macOS/Linux:
        # from repo root
        PYTHONPATH=src python src/dashboard/app.py

    2. Windows (cmd):
        # from repo root
        set FLASK_APP=src/dashboard/app.py&& flask run --port 5000

    FLASK_APP=src/dashboard/app.py flask run --port 5000

    Or:

    python src/dashboard/app.py

Endpoints:
- GET  /                     -> dashboard page
- POST /predict              -> single-record prediction (form or JSON)
- POST /predict_batch        -> CSV upload for batch predictions
- GET  /api/metrics          -> returns model metrics JSON (if present)
- GET  /static/...           -> static assets (Bootstrap/Chart.js loaded locally)
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

import pandas as pd
from flask import (
    Flask,
    abort,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

from dashboard.predict import ModelWrapper, find_model

# Configure app
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_AUTO_RELOAD = True

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB uploads
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "..", "reports", "tmp_uploads")
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-key")  # replace in prod

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "..", "reports", "explain"), exist_ok=True)

# Logging setup
handler = RotatingFileHandler(
    os.path.join(BASE_DIR, "..", "reports", "dashboard.log"),
    maxBytes=5_000_000,
    backupCount=2,
)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)


def load_wrapper(preferred: str | None = None) -> ModelWrapper:
    return ModelWrapper(preferred)


@app.route("/reports/explain/<path:filename>")
def explain_file(filename: str):
    safe_dir = os.path.abspath(
        os.path.join(current_app.root_path, "..", "..", "reports", "explain")
    )
    file_path = os.path.join(safe_dir, filename)

    # Basic safety: ensure the requested file is inside the directory
    if not os.path.commonpath([safe_dir, os.path.abspath(file_path)]) == safe_dir:
        abort(403)
    if not os.path.exists(file_path):
        abort(404)
    return send_from_directory(safe_dir, filename)


@app.route("/", methods=["GET"])
def index():
    # Try to detect a model and metrics
    model_path = find_model()
    metrics = {}
    metrics_file = None
    if model_path:
        base = os.path.basename(model_path)
        model_name = os.path.splitext(base)[0]
        # metrics file typically at reports/<model>_metrics.json
        metrics_file = os.path.join(
            os.path.dirname(os.path.dirname(model_path)),
            "..",
            f"{model_name}_metrics.json",
        )
        if os.path.exists(metrics_file):
            try:
                metrics = pd.read_json(metrics_file).to_dict()
            except Exception:
                try:
                    import json

                    with open(metrics_file) as fh:
                        metrics = json.load(fh)
                except Exception:
                    metrics = {}
    return render_template("index.html", model_path=model_path, metrics=metrics)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept JSON body with a single record (keys matching feature names)
    or form fields (simple UI). Returns JSON with prediction & probability.
    """
    preferred_model = request.form.get("model_path") or request.args.get("model_path")
    data = None
    if request.is_json:
        data = request.get_json()
    else:
        # collect form keys into a dict
        data = {k: v for k, v in request.form.items() if k != "model_path"}
    try:
        wrapper = load_wrapper(preferred_model)
        # attempt to convert to DataFrame-friendly structure
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return jsonify({"ok": False, "error": "Invalid input format"}), 400

        res = wrapper.predict_single(df)
        return jsonify(
            {"ok": True, "result": res, "model_info": wrapper.get_model_info()}
        )
    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({"ok": False, "error": str(e)}), 500


ALLOWED_EXTENSIONS = {"csv"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    Accept a CSV upload with feature columns. Returns a CSV with predictions appended
    and also stores it to reports/tmp_uploads for download.
    """
    if "file" not in request.files:
        flash("No file part")
        return redirect(url_for("index"))
    file = request.files["file"]
    if file.filename == "":
        flash("No selected file")
        return redirect(url_for("index"))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)
        try:
            df = pd.read_csv(save_path)
        except Exception as e:
            flash(f"Uploaded file is not a valid CSV: {e}")
            return redirect(url_for("index"))
        try:
            wrapper = load_wrapper(None)
            out_df = wrapper.predict_batch(df)
            out_path = save_path.replace(".csv", "_predictions.csv")
            out_df.to_csv(out_path, index=False)
            # return a link to download
            return send_file(
                out_path, as_attachment=True, download_name=os.path.basename(out_path)
            )
        except Exception as e:
            app.logger.exception("Batch prediction failed")
            flash(f"Batch prediction failed: {e}")
            return redirect(url_for("index"))
    else:
        flash("Invalid file type. CSV only.")
        return redirect(url_for("index"))


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """
    Return model metrics JSON (if present in reports/)
    """
    model = request.args.get("model")
    metrics_path = None
    if model:
        metrics_path = os.path.join("reports", f"{model}_metrics.json")
    else:
        # try to find any metrics file in reports
        candidates = [p for p in os.listdir("reports") if p.endswith("_metrics.json")]
        metrics_path = os.path.join("reports", candidates[0]) if candidates else None

    if metrics_path and os.path.exists(metrics_path):
        try:
            import json

            with open(metrics_path) as fh:
                return jsonify({"ok": True, "metrics": json.load(fh)})
        except Exception as e:
            app.logger.exception("Failed reading metrics")
            return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": False, "error": "No metrics file found"}), 404


if __name__ == "__main__":
    # quick dev runner
    # optional env var DASHBOARD_MODEL to override model used
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=5000, type=int)
    parser.add_argument(
        "--model", default=None, help="Path to model file to force using"
    )
    args = parser.parse_args()
    if args.model:
        os.environ["DASHBOARD_MODEL"] = args.model
    app.run(host=args.host, port=args.port, debug=True)
