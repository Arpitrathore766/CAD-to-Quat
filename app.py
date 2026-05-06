import os
import sys
import json
import uuid
import subprocess
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "obj_file" not in request.files or "json_file" not in request.files:
        return jsonify({"error": "Both OBJ and JSON files are required"}), 400

    session_id = uuid.uuid4().hex[:8]
    obj_path  = os.path.join(UPLOAD_FOLDER, f"{session_id}.obj")
    json_path = os.path.join(UPLOAD_FOLDER, f"{session_id}.json")

    request.files["obj_file"].save(obj_path)
    request.files["json_file"].save(json_path)

    try:
        from analysis import run_analysis
        edge_text = run_analysis(obj_path, json_path)
        return jsonify({"session_id": session_id, "edge_text": edge_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/compute", methods=["POST"])
def compute():
    data      = request.get_json()
    edge_text = data.get("edge_text", "")
    method    = data.get("method", "direct")
    api_key   = os.environ.get("GEMINI_API_KEY", "")

    try:
        from compute import compute_direct, compute_via_gemini
        if method == "gemini":
            if not api_key:
                return jsonify({"error": "GEMINI_API_KEY not set in .env"}), 500
            vectors = compute_via_gemini(edge_text, api_key)
        else:
            vectors = compute_direct(edge_text)
        return jsonify({"vectors": vectors})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/visualize", methods=["POST"])
def visualize():
    data       = request.get_json()
    session_id = data.get("session_id")
    vectors    = data.get("vectors")

    obj_path     = os.path.join(UPLOAD_FOLDER, f"{session_id}.obj")
    json_path    = os.path.join(UPLOAD_FOLDER, f"{session_id}.json")
    vectors_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_vectors.json")

    if not os.path.exists(obj_path):
        return jsonify({"error": "Session files not found. Please re-upload."}), 400

    with open(vectors_path, "w") as f:
        json.dump(vectors, f)

    subprocess.Popen([sys.executable, "visualizer.py", obj_path, json_path, vectors_path])
    return jsonify({"status": "Visualizer launched"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
