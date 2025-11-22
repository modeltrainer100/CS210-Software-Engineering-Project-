import os
from typing import List

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

import numpy as np
import joblib
from PIL import Image
import io
import tensorflow as tf
import google.generativeai as genai

# -------------------------
# ENV + APP SETUP
# -------------------------
load_dotenv()

app = Flask(__name__)
CORS(app)  # allow all origins for now (fine for project/demo)

# -------------------------
# LOAD MODELS
# -------------------------

# Adjust paths if your files are in a subfolder
CROP_MODEL_PATH = "xgb_pipeline.pkl"
FERT_MODEL_PATH = "randpipe.pkl"
DISEASE_MODEL_PATH = "best_cnn.keras"

try:
  crop_model = joblib.load(CROP_MODEL_PATH)
except Exception as e:
  print("⚠️ Failed to load crop model:", e)
  crop_model = None

try:
  fert_model = joblib.load(FERT_MODEL_PATH)
except Exception as e:
  print("⚠️ Failed to load fertilizer model:", e)
  fert_model = None

try:
  disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH)
except Exception as e:
  print("⚠️ Failed to load disease model:", e)
  disease_model = None

# If you have label encoders, you can load them too
CROP_LABELS_PATH = "crop_label_encoder.pkl"
FERT_LABELS_PATH = "fert_label_encoder.pkl"
DISEASE_LABELS_PATH = "disease_labels.pkl"

def safe_load_pickle(path):
  if not os.path.exists(path):
    return None
  import pickle
  with open(path, "rb") as f:
    return pickle.load(f)

crop_label_encoder = safe_load_pickle(CROP_LABELS_PATH)
fert_label_encoder = safe_load_pickle(FERT_LABELS_PATH)
disease_label_encoder = safe_load_pickle(DISEASE_LABELS_PATH)

# -------------------------
# GEMINI CHAT SETUP
# -------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
  genai.configure(api_key=GEMINI_API_KEY)
  chat_model = genai.GenerativeModel(
    "gemini-1.5-flash",
    system_instruction=(
      "You are DeepAgro AI, an agricultural expert for Indian farmers. "
      "Give clear, practical advice about crops, soil, fertilizer, pests, "
      "and weather-based planning. Be concise and friendly."
    ),
  )
else:
  chat_model = None
  print("⚠️ GEMINI_API_KEY not set. /api/chat will return a fallback reply.")


# -------------------------
# HELPERS
# -------------------------

def validate_fields(data, required: List[str]):
  missing = [f for f in required if f not in data]
  if missing:
    return False, f"Missing fields: {', '.join(missing)}"
  return True, ""


# -------------------------
# CROP PREDICTION API
# -------------------------
@app.route("/api/crop", methods=["POST"])
def api_crop():
  if crop_model is None:
    return jsonify({"error": "Crop model not loaded on server"}), 500

  data = request.get_json() or {}

  required = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
  ok, msg = validate_fields(data, required)
  if not ok:
    return jsonify({"error": msg}), 400

  try:
    x = np.array(
      [[
        float(data["N"]),
        float(data["P"]),
        float(data["K"]),
        float(data["temperature"]),
        float(data["humidity"]),
        float(data["ph"]),
        float(data["rainfall"]),
      ]]
    )
    pred = crop_model.predict(x)[0]

    # decode if label encoder exists
    if crop_label_encoder is not None:
      try:
        pred = crop_label_encoder.inverse_transform([pred])[0]
      except Exception:
        pred = str(pred)

    # if model supports predict_proba, get top 3 crops (optional)
    top3 = []
    if hasattr(crop_model, "predict_proba"):
      probas = crop_model.predict_proba(x)[0]
      indices = np.argsort(probas)[::-1][:3]
      for idx in indices:
        label = idx
        if crop_label_encoder is not None:
          try:
            label = crop_label_encoder.inverse_transform([idx])[0]
          except Exception:
            label = str(idx)
        top3.append({
          "crop": str(label),
          "confidence": float(probas[idx]),
        })

    return jsonify({
      "crop": str(pred),
      "top3": top3,
    })

  except Exception as e:
    print("❌ /api/crop error:", e)
    return jsonify({"error": "Crop prediction failed"}), 500


# -------------------------
# FERTILIZER PREDICTION API
# -------------------------
@app.route("/api/fertilizer", methods=["POST"])
def api_fertilizer():
  if fert_model is None:
    return jsonify({"error": "Fertilizer model not loaded on server"}), 500

  data = request.get_json() or {}

  # adjust fields to match what your model expects
  required = ["temperature", "humidity", "moisture", "N", "P", "K", "ph"]
  ok, msg = validate_fields(data, required)
  if not ok:
    return jsonify({"error": msg}), 400

  try:
    x = np.array(
      [[
        float(data["temperature"]),
        float(data["humidity"]),
        float(data["moisture"]),
        float(data["N"]),
        float(data["P"]),
        float(data["K"]),
        float(data["ph"]),
      ]]
    )

    pred = fert_model.predict(x)[0]

    if fert_label_encoder is not None:
      try:
        pred = fert_label_encoder.inverse_transform([pred])[0]
      except Exception:
        pred = str(pred)

    return jsonify({
      "fertilizer": str(pred),
    })

  except Exception as e:
    print("❌ /api/fertilizer error:", e)
    return jsonify({"error": "Fertilizer prediction failed"}), 500


# -------------------------
# DISEASE DETECTION API
# -------------------------
@app.route("/api/disease", methods=["POST"])
def api_disease():
  if disease_model is None:
    return jsonify({"error": "Disease model not loaded on server"}), 500

  if "image" not in request.files:
    return jsonify({"error": "No image uploaded (field name should be 'image')"}), 400

  file = request.files["image"]

  try:
    img = Image.open(file.stream).convert("RGB")
    # adjust size & preprocessing based on your CNN
    img = img.resize((224, 224))
    arr = np.array(img, dtype="float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = disease_model.predict(arr)[0]
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))

    if disease_label_encoder is not None:
      try:
        disease_name = disease_label_encoder.inverse_transform([idx])[0]
      except Exception:
        disease_name = str(idx)
    else:
      disease_name = str(idx)

    return jsonify({
      "disease": disease_name,
      "confidence": conf,
    })

  except Exception as e:
    print("❌ /api/disease error:", e)
    return jsonify({"error": "Disease prediction failed"}), 500


# -------------------------
# CHATBOT API (Gemini)
# -------------------------
@app.route("/api/chat", methods=["POST"])
def api_chat():
  data = request.get_json() or {}
  message = (data.get("message") or "").strip()
  history = data.get("history") or []

  if not message:
    return jsonify({"reply": "Please ask a farming question."}), 400

  # Fallback if Gemini not configured
  if chat_model is None:
    reply = (
      "DeepAgro AI is not fully connected to Gemini on this server yet. "
      f"You asked: \"{message}\". "
      "Please contact the team to configure GEMINI_API_KEY."
    )
    return jsonify({"reply": reply})

  # Convert frontend history -> Gemini format
  formatted_history = []
  for msg in history:
    role = "user" if msg.get("role") == "user" else "model"
    content = msg.get("content", "")
    formatted_history.append({"role": role, "parts": content})

  try:
    resp = chat_model.generate_content(
      formatted_history + [{"role": "user", "parts": message}],
      generation_config={"temperature": 0.6},
    )
    reply_text = (resp.text or "").strip()
    if not reply_text:
      reply_text = "I couldn't generate a response. Please try rephrasing your question."

    return jsonify({"reply": reply_text})

  except Exception as e:
    print("❌ /api/chat error:", e)
    return jsonify({"reply": "AI service failed. Please try again later."}), 500


if __name__ == "__main__":
  port = int(os.getenv("PORT", "5000"))
  app.run(host="0.0.0.0", port=port, debug=True)
