# 🔍 ToxiScope — Multi-Label Toxicity Classifier

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey)
![NLP](https://img.shields.io/badge/NLP-Transformers-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Problem Statement
Online platforms receive millions of user-generated comments daily. Manual moderation is slow, expensive, and inconsistent. Automated toxicity detection at inference time is essential for maintaining safe communities at scale.

## Solution
ToxiScope is a real-time multi-label toxicity classifier that analyses any input text across **6 harm categories** simultaneously, returning confidence scores for each category with a clean, interactive web UI.

## Categories Detected
| Category       | Description                                  |
|----------------|----------------------------------------------|
| Toxic          | General hateful or abusive language          |
| Severely Toxic | Extreme abuse or hate speech                 |
| Obscene        | Sexually explicit or crude content           |
| Threat         | Direct threats of violence or harm           |
| Insult         | Personal insults and degrading language      |
| Identity Hate  | Hate based on identity (race, gender, etc.)  |

## Outcome
- Instant classification with **per-category probability scores**
- Visual confidence meters and overall risk score
- Clean REST API endpoint (`POST /classify`) usable from any client

## Tech Stack
- **Backend**: Python, Flask
- **NLP**: HuggingFace Transformers (swap to `unitary/toxic-bert` for production)
- **Frontend**: Vanilla JS, custom CSS — no framework bloat

## Quickstart

```bash
# 1. Clone
git clone https://github.com/your-username/toxiscope.git
cd toxiscope

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python app.py

# 4. Open http://localhost:5000
```

## API Usage

```bash
curl -X POST http://localhost:5000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "You are a complete idiot."}'
```

**Response:**
```json
{
  "scores": {
    "toxic": 0.8808,
    "severe_toxic": 0.3775,
    "obscene": 0.2689,
    "threat": 0.1192,
    "insult": 0.8808,
    "identity_hate": 0.1192
  },
  "overall": 0.8808,
  "is_toxic": true,
  "verdict": "⚠️ Potentially Harmful Content Detected"
}
```

## Upgrade: Use a Real Model

To replace the lexicon classifier with `unitary/toxic-bert`:

```python
from transformers import pipeline
classifier = pipeline("text-classification", model="unitary/toxic-bert", return_all_scores=True)

def classify(text):
    results = classifier(text)[0]
    scores = {r['label'].lower(): round(r['score'], 4) for r in results}
    overall = max(scores.values())
    return {
        "scores": scores,
        "overall": overall,
        "is_toxic": overall > 0.55,
        "verdict": "⚠️ Potentially Harmful" if overall > 0.55 else "✅ Appears Safe"
    }
```

## Dataset
Trained models in this space typically use the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset (160k Wikipedia comments, labeled by human raters).


