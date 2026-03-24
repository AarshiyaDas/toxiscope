from flask import Flask, render_template, request, jsonify
import re
import math

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Lightweight rule-based + lexicon classifier (no GPU / heavy model needed)
# Swap this module out for a HuggingFace model in production (see README)
# ---------------------------------------------------------------------------

TOXIC_LEXICON = {
    "toxic":         ["idiot", "stupid", "dumb", "moron", "hate", "kill", "die", "loser", "trash", "garbage", "worthless"],
    "severe_toxic":  ["fuck", "shit", "bitch", "bastard", "damn", "ass", "crap"],
    "obscene":       ["sex", "nude", "porn", "naked", "explicit"],
    "threat":        ["kill you", "hurt you", "destroy you", "end you", "attack", "murder", "threaten"],
    "insult":        ["ugly", "fat", "pathetic", "useless", "failure", "incompetent", "clown", "joke"],
    "identity_hate": ["racist", "sexist", "homophobe", "slur", "discriminate"],
}

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
LABEL_DISPLAY = {
    "toxic":         "Toxic",
    "severe_toxic":  "Severely Toxic",
    "obscene":       "Obscene",
    "threat":        "Threat",
    "insult":        "Insult",
    "identity_hate": "Identity Hate",
}

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def classify(text: str) -> dict:
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    word_set = set(words)

    scores = {}
    for label, keywords in TOXIC_LEXICON.items():
        hits = 0
        for kw in keywords:
            if " " in kw:          # phrase match
                if kw in text_lower:
                    hits += 2
            elif kw in word_set:
                hits += 1
        # map hit count → probability via a scaled sigmoid
        raw = (hits - 0.5) * 2.5
        scores[label] = round(sigmoid(raw), 4)

    overall = max(scores.values())
    is_toxic = overall > 0.55

    return {
        "scores": scores,
        "overall": round(overall, 4),
        "is_toxic": is_toxic,
        "verdict": "⚠️ Potentially Harmful Content Detected" if is_toxic else "✅ Content Appears Safe",
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify_route():
    data = request.get_json()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) > 2000:
        return jsonify({"error": "Text too long (max 2000 chars)"}), 400
    result = classify(text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
