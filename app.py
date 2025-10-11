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

analyzer = SpeechAnalyzer()
db = client["stutter_db"]
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


def analyze_audio_thread(filepath, task_id, audio_bytes):
    try:
        # This part remains the same
        analysis_results = analyzer.analyze_audio_file(filepath)
        
        # --- FIX STARTS HERE ---
        # On success, update the task with results and "completed" status
        tasks_collection.update_one(
            {"task_id": task_id},
            {"$set": {
                "status": "completed",
                "results": analysis_results,
                "updatedAt": datetime.now(timezone.utc).isoformat()
            }}
        )
        print(f"✅ Task {task_id} completed and status updated in DB.")
        # --- FIX ENDS HERE ---

    except Exception as e:
        print(f"❌ Error during analysis for task {task_id}: {e}")
        
        # --- FIX STARTS HERE ---
        # On failure, update the task with an error message and "failed" status
        tasks_collection.update_one(
            {"task_id": task_id},
            {"$set": {
                "status": "failed",
                "error": str(e),
                "updatedAt": datetime.now(timezone.utc).isoformat()
            }}
        )
        print(f"🔥 Task {task_id} failed and status updated in DB.")
        # --- FIX ENDS HERE ---
    finally:
        # Cleanup files
        if os.path.exists(filepath):
            os.remove(filepath)
        if filepath.endswith(".wav"):
            mp4_path = filepath.replace(".wav", ".mp4")
            if os.path.exists(mp4_path):
                os.remove(mp4_path)
        # Clean results only for this task
        task_folder = os.path.join("./results", task_id)
        if os.path.exists(task_folder):
            shutil.rmtree(task_folder)
        # if os.path.exists("./results"):
        #     shutil.rmtree("./results")
        # if os.path.exists("./uploads"):
        #     shutil.rmtree("./uploads")


@app.route("/upload_audio/<task_id>", methods=["POST"])
def upload_audio(task_id):
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # --- THE CRITICAL FIX ---
    # We will ONLY use the filename that the frontend has already sanitized.
    # We use secure_filename as a security best practice.
    filename = secure_filename(file.filename)

    # Create the 'uploads' directory if it doesn't exist.
    os.makedirs("uploads", exist_ok=True)

    # Construct the filepath using ONLY the secure, sanitized filename.
    filepath = os.path.join("uploads", filename)
    
    # Save the file to the server.
    file.save(filepath)

    # Now, proceed with your logic using the original task_id for the database.
    with open(filepath, "rb") as f:
        file_bytes = f.read()

    tasks_collection.insert_one(
        {"task_id": task_id, "status": "processing", "timestamp": datetime.now()}
    )

    # Handle MP4 to WAV conversion if necessary.
    if filename.lower().endswith(".mp4"):
        wav_filepath = os.path.splitext(filepath)[0] + ".wav"
        if not extract_audio(filepath, wav_filepath):
            return jsonify({"error": "Failed to extract audio"}), 500
        filepath = wav_filepath
        with open(filepath, "rb") as f:
            file_bytes = f.read()

    # Start the analysis thread with the original task_id for database tracking.
    thread = threading.Thread(
        target=analyze_audio_thread, args=(filepath, task_id, file_bytes)
    )
    thread.start()

    return jsonify({"message": "Processing started", "task_id": task_id})
# @app.route("/upload_audio", methods=["POST"])
# def upload_audio():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["file"]
#     task_id = datetime.now().strftime("%Y%m%d_%H%M%S")

#     # Create uploads directory if needed
#     os.makedirs("uploads", exist_ok=True)
#     filepath = os.path.join("uploads", f"{task_id}_{secure_filename(file.filename)}")
#     file.save(filepath)
#     file_bytes = open(filepath, "rb").read()

#     # Store initial task record (without filename)
#     tasks_collection.insert_one(
#         {"task_id": task_id, "status": "processing", "timestamp": datetime.now()}
#     )

#     # Handle MP4 files
#     if file.filename.endswith(".mp4"):
#         wav_filepath = filepath.replace(".mp4", ".wav")
#         if not extract_audio(filepath, wav_filepath):
#             return jsonify({"error": "Failed to extract audio"}), 500
#         filepath = wav_filepath
#         file_bytes = open(filepath, "rb").read()

#     # Start processing thread with audio bytes
#     thread = threading.Thread(
#         target=analyze_audio_thread, args=(filepath, task_id, file_bytes)
#     )
#     thread.start()

#     return jsonify({"message": "Processing started", "task_id": task_id})


@app.route("/tasks", methods=["GET"])
def list_tasks():
    """List all task IDs"""
    try:
        tasks = list(
            tasks_collection.find(
                {}, {"_id": 0, "task_id": 1, "status": 1, "timestamp": 1}
            )
        )
        return jsonify({"status": "success", "count": len(tasks), "tasks": tasks})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    

@app.route("/task_status/<task_id>", methods=["GET"])
def task_status(task_id):
    """
    Checks the status of a task. Returns the current status ('processing', 'completed', 'failed')
    or 'not_found' if the task ID does not exist.
    """
    # --- FIX STARTS HERE ---
    task = tasks_collection.find_one({"task_id": task_id})
    
    if task:
        # Prepare response, converting MongoDB's _id for JSON compatibility
        response = {
            "status": task.get("status"),
            "results": task.get("results"),
            "error": task.get("error")
        }
        return jsonify(response)
    else:
        # This will now only happen if the task ID is genuinely wrong
        return jsonify({"status": "not_found", "error": "Task ID not found in the database."}), 404
    # --- FIX ENDS HERE ---


@app.route("/get_result/<task_id>", methods=["GET"])
def get_result(task_id):
    """Retrieve analysis results"""
    task = tasks_collection.find_one({"task_id": task_id}, {"_id": 0})

    if not task:
        return jsonify({"error": "Task not found"}), 404

    if task["status"] != "completed":
        return jsonify({"status": task["status"]}), 202

    # Handle cases where "results" field doesn't exist (old tasks)
    if "results" not in task:
        return jsonify({"error": "Results not available for this task"}), 404

    # DEBUG: Print what's in the results to see what's missing
    results = task["results"]
    print(f"DEBUG - Task {task_id} results keys: {list(results.keys())}")
    if "visualization" in results:
        viz_data = results["visualization"]
        if viz_data:
            print(f"DEBUG - Visualization data length: {len(viz_data)} characters")
        else:
            print("DEBUG - Visualization data is None/empty")
    else:
        print("DEBUG - No 'visualization' key in results")

    return jsonify(results)

# Ensure UTF-8 for console output (Windows fix)
sys.stdout.reconfigure(encoding='utf-8')

# Login Route
@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        user_type = data.get('userType')

        if not email or not password or not user_type:
            return jsonify({"error": "Missing fields"}), 400

        user = users_collection.find_one({"email": email.lower(), "userType": user_type})
        
        if not user:
            return jsonify({"error": "Invalid credentials"}), 401

        stored_password = user['password']
        # Ensure stored password is bytes
        if isinstance(stored_password, str):
            stored_password = stored_password.encode('utf-8')

        if bcrypt.checkpw(password.encode('utf-8'), stored_password):
            # Update last login time
            users_collection.update_one(
                {"_id": user["_id"]},
                {"$set": {"lastLogin": datetime.now(timezone.utc).isoformat()}}
            )
            return jsonify({"message": "Logged in successfully"}), 200
        else:
            return jsonify({"error": "Invalid credentials"}), 401

    except Exception as e:
        print(f"❌ Login Error: {e}")
        return jsonify({"error": "Internal server error"}), 500
    
@app.route("/verify_user", methods=["POST"])
def verify_user():
    try:
        data = request.get_json()
        name = (data.get("name") or "").strip()
        email = (data.get("email") or "").strip().lower()  # normalize email

        if not name or not email:
            return jsonify({"error": "Missing fields"}), 400

        # Match only by email (since names may vary)
        user = users_collection.find_one({"email": email})

        if user:
            return jsonify({
                "userId": str(user["_id"]),
                "name": user["name"]
            }), 200
        else:
            return jsonify({"error": "User not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/user_name/<user_id>", methods=["GET"])
def get_user_name(user_id):
    try:
        user = users_collection.find_one({"_id": ObjectId(user_id)})
        if user:
            return jsonify({"name": user["name"]}), 200
        else:
            return jsonify({"error": "User not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Signup Route
@app.route('/api/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        user_type = data.get('userType')

        if not name or not email or not password or not user_type:
            return jsonify({"error": "Missing fields"}), 400

        # Check if the email already exists
        existing_user = users_collection.find_one({"email": email.lower()})
        if existing_user:
            return jsonify({"error": "Email already in use"}), 400

        # Hash the password (store as bytes in MongoDB)
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Create new user object
        new_user = {
            "name": name,
            "email": email.lower(),
            "password": hashed_password,  # stored as bytes
            "userType": user_type,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "updatedAt": datetime.now(timezone.utc).isoformat(),
            "lastLogin": None,
            "isActive": True
        }

        # Insert the user into the database
        result = users_collection.insert_one(new_user)

        if result.acknowledged:
            return jsonify({"message": "User created successfully"}), 201
        else:
            return jsonify({"error": "Failed to create user"}), 500

    except Exception as e:
        print(f"❌ Signup Error: {e}")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

