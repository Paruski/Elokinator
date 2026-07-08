#!/usr/bin/env python3
"""
server.py — Chess Level Estimator

Sirve 10 posiciones críticas. El usuario elige jugada en cada una.
El modelo GradientBoosting + ajuste Bayesiano estima su banda de nivel.
"""
from __future__ import annotations
import json, pickle, math
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"
TEST_DATA = BASE / "test_data.json"
MODEL = BASE / "best_model.pkl"

app = FastAPI(title="Chess Level Estimator")

# Load test data
with open(TEST_DATA) as f:
    test_data = json.load(f)

BANDS = test_data["bands"]
POSITIONS = test_data["positions"]

# Midpoints of each band for expected rating calculation
BAND_MIDPOINTS = [350, 900, 1300, 1700, 2100, 2475, 2700]

# Load ML model
with open(MODEL, "rb") as f:
    pkg = pickle.load(f)
model = pkg["model"]
scaler = pkg["scaler"]
prior_probs = np.array(pkg["prior_probs"], dtype=np.float64)
posterior_alpha = pkg["posterior_alpha"]
band_idx = {i: b for i, b in enumerate(BANDS)}

pos_lookup = {}
for p in POSITIONS:
    pos_lookup[p["id"]] = p

def posterior_adjust(probs, prior=None, alpha=None):
    """Bayesian adjustment: posterior ∝ likelihood^(1-alpha) * prior^alpha"""
    if prior is None: prior = prior_probs
    if alpha is None: alpha = posterior_alpha
    if alpha <= 0: return probs
    log_lik = np.log(probs + 1e-10)
    log_prior = np.log(prior + 1e-10)
    log_post = (1 - alpha) * log_lik + alpha * log_prior
    log_post = log_post - log_post.max()
    post = np.exp(log_post)
    return post / post.sum()

def predict_band(move_choices: dict[int, str]) -> dict:
    """Predict band from dict of {pos_id: move_uci}."""
    feats = []
    pos_results = []
    seen_positions = 0

    for pid in range(1, 11):
        pdata = pos_lookup[pid]
        uci = move_choices.get(pid)
        if uci and uci in pdata["moves"]:
            cp_loss = pdata["moves"][uci]["cp_loss"]
            feats.append(cp_loss)
            seen_positions += 1
            pos_results.append({
                "id": pid,
                "chosen_move_san": pdata["moves"][uci]["san"],
                "cp_loss": cp_loss,
                "quality": pdata["moves"][uci]["quality"],
            })
        else:
            feats.append(15.0)
            pos_results.append({
                "id": pid, "chosen_move_san": None,
                "cp_loss": None, "quality": None,
            })

    if seen_positions < 3:
        return {"error": "Se necesitan al menos 3 posiciones"}

    X = np.array([feats], dtype=np.float64)
    Xs = scaler.transform(X)

    raw_probs = model.predict_proba(Xs)[0]
    adj_probs = posterior_adjust(raw_probs)
    pred_class = np.argmax(adj_probs)
    pred_band = band_idx[pred_class]

    expected_rating = float(adj_probs @ np.array(BAND_MIDPOINTS, dtype=np.float64))
    confidence = float(adj_probs.max())

    band_probs = {BANDS[i]: float(adj_probs[i]) for i in range(7)}

    played_cps = [f for f in feats[:10] if f != 15.0]
    mean_cp = float(np.mean(played_cps)) if played_cps else 0

    return {
        "predicted_band": pred_band,
        "expected_rating": round(expected_rating),
        "confidence": confidence,
        "band_probabilities": band_probs,
        "mean_cp_loss": round(mean_cp, 1),
        "positions_played": seen_positions,
        "posterior_alpha_used": posterior_alpha,
    }


@app.get("/api/test-data")
def get_test_data():
    return {
        "positions": [{
            "id": p["id"],
            "fen": p["fen"],
            "turn": p["turn"],
            "description": p.get("description", ""),
            "moves": {uci: {"san": info["san"]} for uci, info in p["moves"].items()},
        } for p in POSITIONS],
        "bands": BANDS,
        "total_positions": len(POSITIONS),
    }


class GuessInput(BaseModel):
    moves: dict[str, str]

@app.post("/api/evaluate")
def evaluate(guess: GuessInput):
    move_choices = {int(k): v for k, v in guess.moves.items()}
    result = predict_band(move_choices)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

@app.get("/")
def index():
    return FileResponse(str(STATIC / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
