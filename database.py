# database.py — MongoDB Atlas database setup and helper functions
# Requires: pip install pymongo
# Add to .env: MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/verdant

import os
from datetime import datetime, timezone

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient, DESCENDING
from pymongo.errors import DuplicateKeyError

load_dotenv()

# ── Connect to MongoDB Atlas ──────────────────────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI", "")

if not MONGO_URI:
    raise RuntimeError(
        "MONGO_URI not set. Add it to your .env file:\n"
        "MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/verdant"
    )

_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
_db     = _client.get_default_database()   # uses DB name from URI (verdant)

# Collections
users_col = _db["users"]
scans_col = _db["scans"]

print(f"✓ MongoDB connected → database: '{_db.name}'")


# ── Create indexes on first run ───────────────────────────────────────────────
def init_db():
    """Create unique indexes if they don't exist."""
    users_col.create_index("email",    unique=True)
    users_col.create_index("username", unique=True)
    scans_col.create_index([("user_id", DESCENDING), ("created_at", DESCENDING)])
    print("✓ MongoDB indexes ready")


# ── User helpers ──────────────────────────────────────────────────────────────
def _user_to_dict(doc) -> dict | None:
    """Convert MongoDB document to plain dict with string id."""
    if doc is None:
        return None
    return {
        "id":         str(doc["_id"]),
        "username":   doc["username"],
        "email":      doc["email"],
        "password":   doc["password"],
        "created_at": doc.get("created_at", ""),
    }


def get_user_by_id(user_id: str) -> dict | None:
    try:
        doc = users_col.find_one({"_id": ObjectId(user_id)})
        return _user_to_dict(doc)
    except Exception:
        return None


def get_user_by_email(email: str) -> dict | None:
    doc = users_col.find_one({"email": email.lower()})
    return _user_to_dict(doc)


def get_user_by_username(username: str) -> dict | None:
    doc = users_col.find_one({"username": username})
    return _user_to_dict(doc)


def create_user(username: str, email: str, password_hash: str):
    """
    Insert a new user. Returns (user_dict, error_string).
    error_string is None on success.
    """
    try:
        result = users_col.insert_one({
            "username":   username,
            "email":      email.lower(),
            "password":   password_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        user = get_user_by_id(str(result.inserted_id))
        return user, None

    except DuplicateKeyError as e:
        if "username" in str(e):
            return None, "Username already taken"
        if "email" in str(e):
            return None, "Email already registered"
        return None, "Registration failed"

    except Exception as e:
        return None, str(e)


# ── Scan helpers ──────────────────────────────────────────────────────────────
def save_scan(user_id: str, scan_type: str, image_thumb: str,
              plant_name: str, result: str, confidence: float, advice: str):
    """Save a scan result to MongoDB."""
    scans_col.insert_one({
        "user_id":     user_id,
        "scan_type":   scan_type,         # 'disease' or 'identify'
        "image_thumb": image_thumb,       # base64 JPEG string
        "plant_name":  plant_name,
        "result":      result,
        "confidence":  confidence,
        "advice":      advice,
        "created_at":  datetime.now(timezone.utc).isoformat(),
    })


def get_user_scans(user_id: str, limit: int = 50) -> list:
    """Return the most recent scans for a user as a list of dicts."""
    docs = scans_col.find(
        {"user_id": user_id},
        sort=[("created_at", DESCENDING)],
        limit=limit,
    )
    scans = []
    for doc in docs:
        scans.append({
            "id":          str(doc["_id"]),
            "user_id":     doc["user_id"],
            "scan_type":   doc.get("scan_type", ""),
            "image_thumb": doc.get("image_thumb", ""),
            "plant_name":  doc.get("plant_name", ""),
            "result":      doc.get("result", ""),
            "confidence":  doc.get("confidence", 0),
            "advice":      doc.get("advice", ""),
            "created_at":  doc.get("created_at", "")[:16].replace("T", " "),
        })
    return scans


def delete_scan(scan_id: str, user_id: str):
    """Delete a scan only if it belongs to this user."""
    try:
        scans_col.delete_one({
            "_id":     ObjectId(scan_id),
            "user_id": user_id,
        })
    except Exception:
        pass