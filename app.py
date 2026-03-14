import io
import os

from dotenv import load_dotenv
from flask import (Flask, flash, jsonify, redirect,
                   render_template, request, session, url_for)
from PIL import Image
from werkzeug.security import check_password_hash, generate_password_hash

import database as db
from model_utils import identify_plant, validate_and_diagnose

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "verdant-dev-secret-change-in-production")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB

# ── Init database on startup ─────────────────────────────────────────────────
db.init_db()


# ── Auth helpers ──────────────────────────────────────────────────────────────
def logged_in():
    return "user_id" in session

def current_user():
    if not logged_in():
        return None
    return db.get_user_by_id(session["user_id"])

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not logged_in():
            flash("Please log in to continue.", "info")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ── Context processor — make user available in all templates ─────────────────
@app.context_processor
def inject_user():
    return {"current_user": current_user()}


# ── Routes: Public ────────────────────────────────────────────────────────────
@app.route("/")
def landing():
    if logged_in():
        return redirect(url_for("dashboard"))
    return render_template("landing.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if logged_in():
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm", "")

        # Validation
        errors = []
        if not username or len(username) < 3:
            errors.append("Username must be at least 3 characters.")
        if not email or "@" not in email:
            errors.append("Enter a valid email address.")
        if len(password) < 6:
            errors.append("Password must be at least 6 characters.")
        if password != confirm:
            errors.append("Passwords do not match.")

        if errors:
            for e in errors:
                flash(e, "error")
            return render_template("signup.html",
                                   username=username, email=email)

        user, err = db.create_user(username, email, generate_password_hash(password))
        if err:
            flash(err, "error")
            return render_template("signup.html", username=username, email=email)

        session["user_id"]  = user["id"]
        session["username"] = user["username"]
        flash(f"Welcome to Verdant, {username}!", "success")
        return redirect(url_for("dashboard"))

    return render_template("signup.html", username="", email="")


@app.route("/login", methods=["GET", "POST"])
def login():
    if logged_in():
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = db.get_user_by_email(email)
        if not user or not check_password_hash(user["password"], password):
            flash("Invalid email or password.", "error")
            return render_template("login.html", email=email)

        session["user_id"]  = user["id"]
        session["username"] = user["username"]
        flash(f"Welcome back, {user['username']}!", "success")
        return redirect(url_for("dashboard"))

    return render_template("login.html", email="")


@app.route("/logout")
def logout():
    session.clear()
    flash("You've been logged out.", "info")
    return redirect(url_for("landing"))


# ── Routes: App ───────────────────────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")


@app.route("/history")
@login_required
def history():
    scans = db.get_user_scans(session["user_id"], limit=50)
    return render_template("history.html", scans=scans)


@app.route("/history/delete/<string:scan_id>", methods=["POST"])
@login_required
def delete_scan(scan_id):
    db.delete_scan(scan_id, session["user_id"])
    return redirect(url_for("history"))


# ── API: Diagnose disease ─────────────────────────────────────────────────────
@app.route("/api/diagnose", methods=["POST"])
@login_required
def api_diagnose():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    try:
        img_bytes = file.read()
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        result    = validate_and_diagnose(pil_img)

        # Save to history
        db.save_scan(
            user_id    = session["user_id"],
            scan_type  = "disease",
            image_thumb= result["thumb_b64"],
            plant_name = result.get("plant_name", ""),
            result     = result.get("result", ""),
            confidence = result.get("confidence", 0),
            advice     = result.get("advice", ""),
        )
        return jsonify(result)

    except Exception as e:
        print(f"Diagnose error: {e}")
        return jsonify({"error": str(e)}), 500


# ── API: Identify plant ───────────────────────────────────────────────────────
@app.route("/api/identify", methods=["POST"])
@login_required
def api_identify():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    try:
        img_bytes = file.read()
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        result    = identify_plant(pil_img)

        # Save to history
        db.save_scan(
            user_id    = session["user_id"],
            scan_type  = "identify",
            image_thumb= result["thumb_b64"],
            plant_name = result.get("common_name", ""),
            result     = result.get("common_name", "Unknown"),
            confidence = 0,
            advice     = result.get("care_tips", ""),
        )
        return jsonify(result)

    except Exception as e:
        print(f"Identify error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)