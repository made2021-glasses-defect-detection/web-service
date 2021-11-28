import os

from flask import Flask, request, render_template, url_for, send_from_directory, jsonify, abort
from werkzeug.utils import secure_filename
from redis_storage import RedisStorage

app = Flask(__name__, static_url_path="/static")
app.config['UPLOAD_FOLDER'] = "./uploads"
storage = RedisStorage()

@app.route("/status")
def status():
    return { "status": "ok" }

@app.route("/")
def root():
    return render_template('index.html', image_url=None)

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files['file']
    filename = secure_filename(file.filename)
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(full_path)

    payload = {"path": os.path.abspath(full_path)}
    payload = storage.save(payload)
    storage.publish(RedisStorage.INPUT_VALIDATION, payload)

    return render_template(
        'index.html',
        image_url=url_for('download_file', name=filename),
        uid=payload["uid"],
    )

@app.route("/result/<uid>", methods=["GET"])
def result(uid):
    payload = storage.load(uid)
    if payload:
        del payload["path"]
        return jsonify(payload)
    else:
         abort(404)

@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)
