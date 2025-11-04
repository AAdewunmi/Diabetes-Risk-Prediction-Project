#!/usr/bin/env python3
"""
src/dashboard/app.py

Flask dashboard application entrypoint.

Key endpoints:
- GET  /                     -> dashboard page
- POST /predict              -> single-record prediction (JSON or form)
- POST /predict_batch        -> CSV upload (multipart/form-data), returns JSON
- GET  /api_explain_files    -> JSON list of files in reports/explain
- GET  /reports/explain/<f>  -> serve files from reports/explain securely

Run:
  PYTHONPATH=src flask --app src/dashboard/app.py run
  or
  PYTHONPATH=src python src/dashboard/app.py
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional

import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

# Local imports (package-relative; imports kept at top for ruff E402)
from .predict import (
    REPORTS_EXPLAIN_DIR,
    ModelWrapper,
    find_model,
    list_explain_files,
)

# ----- Flask app & paths -----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../src/dashboard
_SRC_ROOT = os.path.dirname(_THIS_DIR)  # .../src

TEMPLATE_DIR = os.path.join(_SRC_ROOT, "templates")
STATIC_DIR = os.path.join(_SRC_ROOT, "static")
REPORTS_DIR = os.path.abspath(os.path.join(_SRC_ROOT, "..", "reports"))
UPLOAD_TMP = os.path.join(REPORTS_DIR, "tmp_uploads")

os.makedirs(UPLOAD_TMP, exist_ok=True)
os.makedirs(REPORTS_EXPLAIN_DIR, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB
app.config["UPLOAD_FOLDER"] = UPLOAD_TMP
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-key")

# Logging
log_path = os.path.join(REPORTS_DIR, "dashboard.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=2)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)


def load_wrapper(preferred: Optional[str] = None) -> ModelWrapper:
    """Factory to build a ModelWrapper with optional preferred model path."""
    return ModelWrapper(preferred)


# ---------------------- Explain files ----------------------


@app.get("/api_explain_files")
def api_explain_files() -> Any:
    """
    Return JSON with the list of explain files and the latest file (if any).

    {
      "ok": True,
      "files": [{"filename": "...", "mtime": <int>}, ...],
      "latest": {"filename": "...", "mtime": <int>} | None
    }
    """
    files = list_explain_files()
    latest = files[0] if files else None
    return jsonify({"ok": True, "files": files, "latest": latest})


@app.get("/reports/explain/<path:filename>")
def explain_file(filename: str):
    """
    Securely serve files from reports/explain without copying them into static.
    """
    safe_dir = REPORTS_EXPLAIN_DIR
    requested = os.path.abspath(os.path.join(safe_dir, filename))

    # Prevent directory traversal
    if os.path.commonpath([safe_dir, requested]) != safe_dir:
        app.logger.warning("Forbidden file access attempt: %s", requested)
        abort(403)
    if not os.path.exists(requested):
        abort(404)
    return send_from_directory(safe_dir, filename)


# ---------------------- Dashboard ----------------------


@app.get("/")
def index():
    """
    Render the main dashboard UI. We attempt to detect a model and any metrics
    (if your template needs them later).
    """
    model_path = find_model()
    # Kept lightweight; template currently only needs the base page
    return render_template("index.html", model_path=model_path or "", metrics={})


# ---------------------- Predictions ----------------------


@app.post("/predict")
def predict():
    """
    Single-record prediction. Accepts form-encoded or JSON payload.
    Returns:
      {
        "ok": True/False,
        "result": {
          "prediction": int,
          "probability": float|None,
          "user_message": str,
          "explanation_files": [{"filename","mtime"}, ...]
        },
        "model_info": {...}
      }
    """
    preferred_model = request.form.get("model_path") or request.args.get("model_path")
    if request.is_json:
        data: Dict[str, Any] = request.get_json()  # type: ignore[assignment]
    else:
        data = {k: v for k, v in request.form.items() if k != "model_path"}

    try:
        wrapper = load_wrapper(preferred_model)
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


@app.post("/predict_batch")
def predict_batch_route():
    """
    Flask route to accept a CSV upload and return batch predictions as JSON.
    Keeps route at /predict_batch so existing clients/tests are unaffected.
    """
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"ok": False, "error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        try:
            df = pd.read_csv(save_path)
        except Exception as e:
            return (
                jsonify(
                    {"ok": False, "error": f"Uploaded file is not a valid CSV: {e}"}
                ),
                400,
            )

        try:
            wrapper = load_wrapper(None)
            res = wrapper.predict_batch(df)  # dict with DataFrame in res['predictions']
            preds_df = res.get("predictions")
            preds_records = (
                preds_df.to_dict(orient="records") if preds_df is not None else []
            )

            result = {
                "n_rows": res.get("n_rows"),
                "mean_probability": res.get("mean_probability"),
                "histogram": res.get("histogram"),
                "explanation_files": res.get("explanation_files", []),
                "predictions": preds_records,
            }
            return jsonify(
                {"ok": True, "result": result, "model_info": wrapper.get_model_info()}
            )
        except Exception as e:
            app.logger.exception("Batch prediction failed")
            return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({"ok": False, "error": "Invalid file type. CSV only."}), 400


if __name__ == "__main__":
    # Minimal dev runner (flask run recommended)
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=host, port=port, debug=True)
