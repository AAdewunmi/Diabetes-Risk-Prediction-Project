#!/usr/bin/env python3
"""
src/dashboard/app.py

Flask dashboard application entrypoint.

Key endpoints:
- GET  /                     -> dashboard page
- POST /predict              -> single-record prediction (JSON or form)
- POST /predict_batch        -> CSV upload (multipart/form-data)
- GET  /api/explain_files    -> JSON list of files in reports/explain
- GET  /reports/explain/<f>  -> serve files from reports/explain securely

Run:
  PYTHONPATH=src flask --app src/dashboard/app.py run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from logging.handlers import RotatingFileHandler

import pandas as pd
from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_from_directory,
)
from werkzeug.utils import secure_filename

from dashboard.predict import ModelWrapper, find_model, list_explain_files

# ensure src import works
_this_dir = os.path.dirname(os.path.abspath(__file__))  # .../src/dashboard
_src_root = os.path.dirname(_this_dir)  # .../src
if _src_root not in os.sys.path:
    os.sys.path.insert(0, _src_root)
# Ensure server-side plotting uses Agg (lowest-risk backend for production/dev)
os.environ.setdefault("MPLBACKEND", "Agg")


BASE_REPO = os.path.abspath(os.path.join(_src_root, ".."))
REPORTS_EXPLAIN_DIR = os.path.join(BASE_REPO, "reports", "explain")
UPLOAD_TMP = os.path.join(BASE_REPO, "reports", "tmp_uploads")
os.makedirs(REPORTS_EXPLAIN_DIR, exist_ok=True)
os.makedirs(UPLOAD_TMP, exist_ok=True)

app = Flask(
    __name__,
    template_folder=os.path.join(_src_root, "templates"),
    static_folder=os.path.join(_src_root, "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = UPLOAD_TMP
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-key")


# helper
def load_wrapper(model_path=None):
    """Utility to consistently load ModelWrapper."""
    return ModelWrapper(model_path)


# logging
log_path = os.path.abspath(os.path.join(_src_root, "..", "reports", "dashboard.log"))
os.makedirs(os.path.dirname(log_path), exist_ok=True)
handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=2)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)


@app.route("/", methods=["GET"])
def index():
    model_path = find_model()
    metrics = {}
    if model_path:
        # try to find metrics file next to reports, e.g. reports/<model>_metrics.json
        base = os.path.basename(model_path)
        name = os.path.splitext(base)[0]
        candidates = [
            os.path.join(os.path.dirname(model_path), "..", f"{name}_metrics.json"),
            os.path.join("reports", f"{name}_metrics.json"),
        ]
        for p in candidates:
            try:
                p_abs = os.path.abspath(p)
                if os.path.exists(p_abs):
                    with open(p_abs, "r") as fh:
                        metrics = json.load(fh)
                    break
            except Exception:
                app.logger.exception("Failed reading metrics file: %s", p)
    return render_template("index.html", model_path=model_path, metrics=metrics)


@app.route("/reports/explain/<path:filename>")
def explain_file(filename: str):
    safe_dir = REPORTS_EXPLAIN_DIR
    requested = os.path.abspath(os.path.join(safe_dir, filename))
    # directory traversal protection
    if not os.path.commonpath([safe_dir, requested]) == safe_dir:
        abort(403)
    if not os.path.exists(requested):
        abort(404)
    return send_from_directory(safe_dir, filename)


@app.route("/api/explain_files", methods=["GET"])
def api_explain_files():
    try:
        files = list_explain_files()
        return jsonify(
            {"ok": True, "files": files, "latest": files[0] if files else None}
        )
    except Exception as e:
        app.logger.exception("Failed to list explain files")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept JSON or form data for a single record.
    Returns: { ok: True, result: { prediction, probability, user_message, explanation_files[...] }, model_info: {...} }
    """
    if request.is_json:
        data = request.get_json()
    else:
        data = {k: v for k, v in request.form.items()}

    try:
        wrapper = ModelWrapper(os.environ.get("DASHBOARD_MODEL"))
    except Exception as e:
        app.logger.exception("Model load failed")
        return jsonify({"ok": False, "error": "Model loading failed: " + str(e)}), 500

    try:
        # coerce to DataFrame and prepare with expected features
        df = pd.DataFrame([data])
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


# 📍 Replace ONLY the /predict_batch route function with the version below
# (keeps the URL the same and returns histogram + JSON-serializable predictions)


@app.route("/predict_batch", methods=["POST"])
def predict_batch_route():
    """
    Accept CSV upload and return JSON:
    {
      ok: True,
      result: {
        n_rows: int,
        mean_probability: float | null,
        histogram: { bin_edges: [..], counts: [..], n_valid: int },
        predictions: [{...}]   # records with 'prediction' and 'probability'
      },
      model_info: {...}
    }
    """
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"ok": False, "error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"ok": False, "error": "Invalid file type. CSV only."}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file.save(save_path)

    try:
        df = pd.read_csv(save_path)
    except Exception as e:
        return (
            jsonify({"ok": False, "error": f"Uploaded file is not a valid CSV: {e}"}),
            400,
        )

    try:
        wrapper = load_wrapper(None)
        res = wrapper.predict_batch(df)  # dict with DataFrame in res['predictions']

        preds_df = res.get("predictions")
        preds_records = (
            preds_df.to_dict(orient="records") if preds_df is not None else []
        )

        # Return JSON with histogram for the client-side chart
        return (
            jsonify(
                {
                    "ok": True,
                    "result": {
                        "n_rows": res.get("n_rows"),
                        "mean_probability": res.get("mean_probability"),
                        "histogram": res.get(
                            "histogram", {"bin_edges": [], "counts": [], "n_valid": 0}
                        ),
                        "predictions": preds_records,
                    },
                    "model_info": wrapper.get_model_info(),
                }
            ),
            200,
        )

    except Exception as e:
        app.logger.exception("Batch prediction failed")
        return jsonify({"ok": False, "error": f"Batch prediction failed: {e}"}), 500


if __name__ == "__main__":
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
