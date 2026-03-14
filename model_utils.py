# model_utils.py — PyTorch model loading and Gemini Vision API calls

import io
import json
import math
import os
import base64

import requests
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# ── Config ───────────────────────────────────────────────────────────────────
CHECKPOINT  = os.path.join(os.path.dirname(__file__), "outputs", "plant_disease_model.pth")
CLASS_INFO  = os.path.join(os.path.dirname(__file__), "outputs", "class_info.json")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE    = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"


# ── Load class info ───────────────────────────────────────────────────────────
with open(CLASS_INFO) as f:
    _info = json.load(f)
CLASS_NAMES = _info["class_names"]
NUM_CLASSES  = _info["num_classes"]

SUPPORTED_PLANTS = sorted(set(
    n.split("___")[0].replace("_", " ") for n in CLASS_NAMES
))
SUPPORTED_PLANTS_STR = ", ".join(SUPPORTED_PLANTS)


# ── Build model ───────────────────────────────────────────────────────────────
def _build_efficientnet(num_classes):
    m = models.efficientnet_b0(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3, inplace=True),
        torch.nn.Linear(in_features, num_classes),
    )
    return m


_model = _build_efficientnet(NUM_CLASSES)
_model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
_model.to(DEVICE)
_model.eval()
print(f"✓ PyTorch model ready on {DEVICE} — {NUM_CLASSES} classes")

_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Make thumbnail ────────────────────────────────────────────────────────────
def make_thumbnail(pil_img, size=400) -> str:
    """Return base64 JPEG thumbnail string."""
    thumb = pil_img.copy()
    thumb.thumbnail((size, size))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


# ── PyTorch disease prediction ────────────────────────────────────────────────
def run_disease_model(pil_img, top_k=5) -> list:
    tensor = _transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(_model(tensor), dim=1)[0]
    top_probs, top_idx = torch.topk(probs, k=min(top_k, NUM_CLASSES))
    return [
        {
            "class":       CLASS_NAMES[i.item()],
            "probability": round(p.item() * 100, 2),
            "plant":       CLASS_NAMES[i.item()].split("___")[0].replace("_", " "),
            "condition":   (CLASS_NAMES[i.item()].split("___")[1].replace("_", " ")
                            if "___" in CLASS_NAMES[i.item()]
                            else CLASS_NAMES[i.item()].replace("_", " ")),
        }
        for p, i in zip(top_probs, top_idx)
    ]


# ── Gemini helper ─────────────────────────────────────────────────────────────
def _call_gemini(image_b64: str, prompt: str, max_tokens: int = 1024) -> dict | None:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    try:
        resp = requests.post(
            f"{GEMINI_URL}?key={api_key}",
            json={
                "contents": [{
                    "parts": [
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                        {"text": prompt},
                    ]
                }],
                "generationConfig": {
                    "temperature":        0.1,
                    "maxOutputTokens":    max_tokens,
                    "responseMimeType":   "application/json",
                },
            },
            timeout=25,
        )
        data = resp.json()
        if "error" in data:
            print(f"  Gemini API error: {data['error'].get('message','')}")
            return None
        candidates = data.get("candidates", [])
        if not candidates:
            print("  Gemini: no candidates returned")
            return None
        raw = candidates[0]["content"]["parts"][0]["text"].strip()

        # ── Robust JSON extraction ──
        # Strip markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()

        # Try direct parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Try extracting just the JSON object between first { and last }
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass

        # Last resort: use regex to extract key-value pairs manually
        import re
        result = {}
        bool_map = {"true": True, "false": False, "null": None}

        # Extract string values:  "key": "value"
        for m in re.finditer(r'"(\w+)"\s*:\s*"([^"]*)"', raw):
            result[m.group(1)] = m.group(2)

        # Extract bool/null values:  "key": true/false/null
        for m in re.finditer(r'"(\w+)"\s*:\s*(true|false|null)', raw):
            result[m.group(1)] = bool_map[m.group(2)]

        if result:
            print(f"  Gemini: used regex fallback parser, got keys: {list(result.keys())}")
            return result

        print(f"  Gemini: could not parse response: {raw[:200]}")
        return None

    except Exception as e:
        print(f"  Gemini exception: {e}")
        return None


# ── Feature 1: Validate + Disease Detection ───────────────────────────────────
def validate_and_diagnose(pil_img) -> dict:
    """
    Step 1 — Ask Gemini if the image is a supported plant leaf.
    Step 2 — If yes, run our PyTorch model.
    Returns a unified result dict.
    """
    thumb_b64 = make_thumbnail(pil_img)

    validate_prompt = f"""You are a plant pathology expert.

Our disease detection model supports ONLY these plants:
{SUPPORTED_PLANTS_STR}

Look at the image and respond in JSON:
{{
  "is_plant": true or false,
  "is_leaf": true or false,
  "plant_name": "plant species name or empty string",
  "in_supported_list": true or false,
  "what_i_see": "one sentence describing the image",
  "advice": "one sentence. If not a plant: ask to upload a leaf. If plant not supported: say model doesn't support it yet and name one common disease to watch for. If supported but blurry: ask for clearer photo."
}}
Only set in_supported_list=true if the plant EXACTLY matches our supported list."""

    parsed = _call_gemini(thumb_b64, validate_prompt, max_tokens=512)

    if parsed is None:
        # Gemini unavailable — run model directly
        preds = run_disease_model(pil_img)
        top   = preds[0]
        return {
            "mode":        "disease",
            "supported":   True,
            "predictions": preds,
            "plant_name":  top["plant"],
            "result":      top["condition"],
            "confidence":  top["probability"],
            "advice":      "",
            "what_i_see":  "",
            "thumb_b64":   thumb_b64,
        }

    in_supported = bool(parsed.get("in_supported_list") and parsed.get("is_leaf"))

    if in_supported:
        preds = run_disease_model(pil_img)
        top   = preds[0]
        return {
            "mode":        "disease",
            "supported":   True,
            "predictions": preds,
            "plant_name":  top["plant"],
            "result":      top["condition"],
            "confidence":  top["probability"],
            "advice":      parsed.get("advice", ""),
            "what_i_see":  parsed.get("what_i_see", ""),
            "thumb_b64":   thumb_b64,
        }
    else:
        return {
            "mode":        "disease",
            "supported":   False,
            "predictions": [],
            "plant_name":  parsed.get("plant_name", ""),
            "result":      "Not supported",
            "confidence":  0,
            "advice":      parsed.get("advice", "This plant is not in our dataset."),
            "what_i_see":  parsed.get("what_i_see", ""),
            "thumb_b64":   thumb_b64,
        }


# ── Feature 2: Plant Identification ──────────────────────────────────────────
def identify_plant(pil_img) -> dict:
    """
    Identify what plant is in the image — doesn't need to be in our dataset.
    Uses Gemini Vision only (no PyTorch model needed).
    """
    thumb_b64 = make_thumbnail(pil_img)

    identify_prompt = """You are a botanist and plant expert.

Look at this image and identify the plant. Respond in JSON:
{
  "is_plant": true or false,
  "common_name": "common name of the plant e.g. Mango, Rose, Tomato",
  "scientific_name": "scientific name e.g. Mangifera indica",
  "plant_family": "plant family e.g. Anacardiaceae",
  "confidence": "High / Medium / Low",
  "description": "2 sentences about this plant — where it grows and what it's used for",
  "care_tips": "2 sentences of care and disease prevention tips specific to this plant",
  "fun_fact": "one interesting fact about this plant"
}
If it is NOT a plant, set is_plant to false and set all other fields to empty strings."""

    parsed = _call_gemini(thumb_b64, identify_prompt, max_tokens=800)

    if parsed is None:
        return {
            "mode":            "identify",
            "is_plant":        None,
            "common_name":     "",
            "scientific_name": "",
            "plant_family":    "",
            "confidence":      "",
            "description":     "",
            "care_tips":       "Gemini API unavailable. Add GEMINI_API_KEY to .env file.",
            "fun_fact":        "",
            "thumb_b64":       thumb_b64,
        }

    return {
        "mode":            "identify",
        "is_plant":        parsed.get("is_plant", False),
        "common_name":     parsed.get("common_name", "Unknown"),
        "scientific_name": parsed.get("scientific_name", ""),
        "plant_family":    parsed.get("plant_family", ""),
        "confidence":      parsed.get("confidence", ""),
        "description":     parsed.get("description", ""),
        "care_tips":       parsed.get("care_tips", ""),
        "fun_fact":        parsed.get("fun_fact", ""),
        "thumb_b64":       thumb_b64,
    }