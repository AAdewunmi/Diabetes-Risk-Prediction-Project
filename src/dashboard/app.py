#!/usr/bin/env python3
"""
src/dashboard/app.py

Flask dashboard application entrypoint.

This file exposes the dashboard and includes a secure route to serve explanation
artifacts from the project's reports/explain directory in-place (no copying).

Run (development):
    PYTHONPATH=src flask --app src/dashboard/app.py run
    or
    PYTHONPATH=src python src/dashboard/app.py

Routes:
- GET  /                                -> dashboard index
- POST /predict                         -> single prediction (form or JSON)
- POST /predict_batch                   -> batch CSV upload
- GET  /reports/explain/<path:filename> -> serve files from reports/explain (secure)
- GET  /api/explain_files               -> list explain files (filename + mtime)
- GET  /api/metrics                     -> JSON metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List

import pandas as pd
from flask import (
    Flask,
    abort,
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

# Ensure src package imports work when running this file directly.
_this_dir = os.path.dirname(os.path.abspath(__file__))  # .../project/src/dashboard
_src_root = os.path.dirname(_this_dir)  # .../project/src
if _src_root not in os.sys.path:
    os.sys.path.insert(0, _src_root)

from dashboard.predict import ModelWrapper, find_model  # noqa: E402

BASE_DIR = _this_dir
REPORTS_EXPLAIN_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..", "reports", "explain")
)
UPLOAD_TMP = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..", "reports", "tmp_uploads")
)

os.makedirs(UPLOAD_TMP, exist_ok=True)
os.makedirs(REPORTS_EXPLAIN_DIR, exist_ok=True)

# point Flask at repo-level template dir (src/templates)
TEMPLATE_DIR = os.path.join(_src_root, "templates")  # _src_root is .../project/src
app = Flask(
    __name__,
    template_folder=os.path.join(_src_root, "templates"),
    static_folder=os.path.join(_src_root, "static"),
)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = UPLOAD_TMP
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-key")

# logging
log_path = os.path.abspath(os.path.join(_src_root, "..", "reports", "dashboard.log"))
os.makedirs(os.path.dirname(log_path), exist_ok=True)
handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=2)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)


def load_wrapper(preferred: str | None = None) -> ModelWrapper:
    return ModelWrapper(preferred)


@app.route("/reports/explain/<path:filename>")
def explain_file(filename: str):
    """
    Securely serve files from reports/explain without copying them into src/static.

    - Validates that the final absolute path is inside REPORTS_EXPLAIN_DIR.
    - Returns 404 if file missing, 403 if attempt to escape directory.
    """
    safe_dir = REPORTS_EXPLAIN_DIR
    requested = os.path.abspath(os.path.join(safe_dir, filename))

    # Prevent directory traversal
    if not os.path.commonpath([safe_dir, requested]) == safe_dir:
        app.logger.warning("Forbidden file access attempt: %s", requested)
        abort(403)
    if not os.path.exists(requested):
        abort(404)
    return send_from_directory(safe_dir, filename)


def _list_explain_files(extensions: List[str] = None) -> List[Dict[str, Any]]:
    """
    Return a list of files in REPORTS_EXPLAIN_DIR filtered by extensions
    with their modification times (as integer timestamps).
    """
    if extensions is None:
        extensions = [".png", ".jpg", ".jpeg", ".gif", ".svg", ".html"]
    out: List[Dict[str, Any]] = []
    try:
        for name in os.listdir(REPORTS_EXPLAIN_DIR):
            path = os.path.join(REPORTS_EXPLAIN_DIR, name)
            if not os.path.isfile(path):
                continue
            if extensions and not any(name.lower().endswith(ext) for ext in extensions):
                continue
            try:
                mtime = int(os.path.getmtime(path))
            except Exception:
                mtime = 0
            out.append({"filename": name, "mtime": mtime})
        out.sort(key=lambda r: r["mtime"], reverse=True)
    except FileNotFoundError:
        return []
    except Exception:
        app.logger.exception("Error listing explain files")
    return out


@app.route("/api/explain_files", methods=["GET"])
def api_explain_files():
    """
    Return JSON with the list of explain files and the latest file (if any).

    Example response:
    {
      "ok": True,
      "files": [{"filename":"shap_summary.png","mtime":163...}, ...],
      "latest": {"filename":"shap_summary.png","mtime":163...} or null
    }
    """
    files = _list_explain_files()
    latest = files[0] if files else None
    return jsonify({"ok": True, "files": files, "latest": latest})


@app.route("/", methods=["GET"])
def index():
    # Try to detect a model and metrics
    model_path = find_model()
    metrics = {}
    if model_path:
        # try to find a metrics file in reports (convention: <model>_metrics.json)
        base = os.path.basename(model_path)
        name = os.path.splitext(base)[0]
        # metrics in reports/ (sibling of reports/models)
        candidates = [
            os.path.join(
                os.path.dirname(os.path.dirname(model_path)),
                "..",
                f"{name}_metrics.json",
            ),
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


@app.route("/predict", methods=["POST"])
def predict():
    preferred_model = request.form.get("model_path") or request.args.get("model_path")
    data = None
    if request.is_json:
        data = request.get_json()
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


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
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
    model = request.args.get("model")
    metrics_path = None
    if model:
        metrics_path = os.path.join("reports", f"{model}_metrics.json")
    else:
        candidates = (
            [p for p in os.listdir("reports") if p.endswith("_metrics.json")]
            if os.path.isdir("reports")
            else []
        )
        metrics_path = os.path.join("reports", candidates[0]) if candidates else None

    if metrics_path and os.path.exists(metrics_path):
        try:
            with open(metrics_path) as fh:
                return jsonify({"ok": True, "metrics": json.load(fh)})
        except Exception as e:
            app.logger.exception("Failed reading metrics")
            return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": False, "error": "No metrics file found"}), 404


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
