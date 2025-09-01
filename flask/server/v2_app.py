import os
from urllib.parse import urlparse
from flask import Flask, request, jsonify, send_from_directory, abort
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 📂 업로드 폴더 (절대 경로)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # 폴더 없으면 자동 생성
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

BLOCKED_PATTERNS = ["php", "asp", "cgi", "wls_internal", "fsms"]

@app.before_request
def block_patterns():
    path = request.path.lower()
    for p in BLOCKED_PATTERNS:
        if p in path:
            abort(403)

# 1) 서버 동작 체크
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask server is running on EC2!"})


# 2) 파일 업로드
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if not file or file.filename.strip() == "":
        return jsonify({"error": "No selected file"}), 400

    safe_name = secure_filename(file.filename)
    if not safe_name:
        return jsonify({"error": "Invalid filename"}), 400

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    try:
        file.save(save_path)
    except Exception as e:
        return jsonify({"error": f"Save failed: {e}"}), 500

    file_url = request.host_url.rstrip("/") + "/uploads/" + safe_name

    # ✅ JSP/Java 쪽과 100% 호환되는 포맷 유지
    return jsonify({
        "message": "File uploaded successfully",
        "url": file_url
    }), 201


# 3) 업로드된 파일 서빙
@app.route("/uploads/<path:filename>", methods=["GET"])
def uploaded_file(filename):
    safe_name = secure_filename(filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    if not os.path.isfile(file_path):
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(app.config["UPLOAD_FOLDER"], safe_name)


# 4) JSP에서 외부 URL을 다운로드해 서버에 저장
@app.route("/download", methods=["POST"])
def download_file():
    data = request.get_json(silent=True) or {}
    file_url = data.get("url")
    if not file_url:
        return jsonify({"error": "No URL provided"}), 400

    # URL 경로에서 파일명 추출
    path = urlparse(file_url).path
    basename = os.path.basename(path)
    safe_name = secure_filename(basename) or "downloaded_file"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)

    try:
        import requests
        r = requests.get(file_url, timeout=15, stream=True)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    except Exception as e:
        return jsonify({"error": f"Download failed: {e}"}), 502

    file_url = request.host_url.rstrip("/") + "/uploads/" + safe_name
    return jsonify({
        "message": "File downloaded and saved",
        "url": file_url
    }), 201


# 5) 파일 삭제 (DELETE /uploads/<filename>)
@app.route("/uploads/<path:filename>", methods=["DELETE"])
def delete_file(filename):
    safe_name = secure_filename(filename)
    if not safe_name:
        return jsonify({"error": "Invalid filename"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    if not os.path.isfile(file_path):
        return jsonify({"error": "File not found"}), 404

    try:
        os.remove(file_path)
    except Exception as e:
        return jsonify({"error": f"Delete failed: {e}"}), 500

    return jsonify({"message": f"{safe_name} deleted."}), 200


# ── 항상 JSON 에러 반환 ──
@app.errorhandler(404)
def _404(_e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(405)
def _405(_e):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(413)
def _413(_e):
    return jsonify({"error": "Payload too large"}), 413

@app.errorhandler(500)
def _500(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

