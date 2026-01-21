from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import threading
import ffmpeg
import shutil
import base64
import bcrypt
import sys
from bson.objectid import ObjectId
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from datetime import datetime,timezone
from pymongo import MongoClient
from bson.binary import Binary
from analyzer import SpeechAnalyzer

app = Flask(__name__)
CORS(app)
load_dotenv()

# MongoDB setup
client = MongoClient("mongodb+srv://admin:admin@stutterdb.hdvzwzs.mongodb.net/stutter_db?retryWrites=true&w=majority&appName=stutterdb")
db = client.stutter_db
tasks_collection = db["tasks"]
users_collection = db["users"]

def extract_audio(mp4_filepath, wav_filepath):
    try:
        ffmpeg.input(mp4_filepath).output(
            wav_filepath, format="wav", acodec="pcm_s16le", ar="16000"
        ).run(overwrite_output=True, quiet=True)
        return True
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return False

def analyze_audio_thread(filepath, task_id, audio_bytes, language="en"):
    try:
        current_analyzer = SpeechAnalyzer(language=language)
        analysis_results = current_analyzer.analyze_audio_file(filepath, language=language)
        
        tasks_collection.update_one(
            {"task_id": task_id},
            {"$set": {
                "status": "completed",
                "results": analysis_results,
                "updatedAt": datetime.now(timezone.utc).isoformat()
            }}
        )
        print(f"✅ Task {task_id} completed for language {language}.")

    except Exception as e:
        print(f"❌ Error during analysis for task {task_id}: {e}")
        tasks_collection.update_one(
            {"task_id": task_id},
            {"$set": {
                "status": "failed",
                "error": str(e),
                "updatedAt": datetime.now(timezone.utc).isoformat()
            }}
        )
    finally:
        if os.path.exists(filepath): os.remove(filepath)

@app.route("/upload_audio/<task_id>", methods=["POST"])
def upload_audio(task_id):
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    language = request.form.get("language", "en")
    if language not in ["en", "hi", "mr"]:
        return jsonify({"error": f"Unsupported language: {language}"}), 400

    filename = secure_filename(file.filename)
    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", filename)
    file.save(filepath)

    with open(filepath, "rb") as f: file_bytes = f.read()

    tasks_collection.insert_one(
        {"task_id": task_id, "status": "processing", "timestamp": datetime.now(), "language": language}
    )

    if filename.lower().endswith(".mp4"):
        wav_filepath = os.path.splitext(filepath)[0] + ".wav"
        if not extract_audio(filepath, wav_filepath):
            return jsonify({"error": "Failed to extract audio"}), 500
        filepath = wav_filepath
        with open(filepath, "rb") as f: file_bytes = f.read()

    thread = threading.Thread(target=analyze_audio_thread, args=(filepath, task_id, file_bytes, language))
    thread.start()
    return jsonify({"message": "Processing started", "task_id": task_id})

@app.route("/task_status/<task_id>", methods=["GET"])
def task_status(task_id):
    task = tasks_collection.find_one({"task_id": task_id})
    if task:
        return jsonify({"status": task.get("status"), "results": task.get("results"), "error": task.get("error")})
    return jsonify({"status": "not_found"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
