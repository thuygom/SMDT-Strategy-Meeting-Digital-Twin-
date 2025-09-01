import os
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # ✅ 폴더 없으면 자동 생성
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 1. 접속 체크
@app.route("/", methods=["GET"])
def home():
    return {"message": "Flask server is running on EC2!"}

# 2. 파일 업로드
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    # ✅ 웹에서 접근 가능한 URL 반환
    file_url = f"http://{request.host}/uploads/{file.filename}"

    return jsonify({
        "message": "File uploaded successfully",
        "url": file_url   # 🔑 Java가 인식할 수 있도록 url 키 추가
    })

# 3. 업로드된 파일 서빙
@app.route("/uploads/<filename>", methods=["GET"])
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# 4. JSP 쪽에서 특정 URL을 가져와서 저장 (옵션)
@app.route("/download", methods=["POST"])
def download_file():
    file_url = request.json.get("url")
    if not file_url:
        return jsonify({"error": "No URL provided"}), 400

    filename = os.path.basename(file_url)
    save_path = os.path.join(UPLOAD_FOLDER, filename)

    import requests
    r = requests.get(file_url)
    with open(save_path, "wb") as f:
        f.write(r.content)

    return jsonify({
        "message": "File downloaded and saved",
        "path": save_path
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


