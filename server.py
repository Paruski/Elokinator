#!/usr/bin/env python3
"""
server.py — Chess Level Estimator (Elokinator)

Sirve el frontend estático y el modelo para inferencia 100% JS.
"""
from __future__ import annotations
import json, pickle
from pathlib import Path
import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"

app = FastAPI(title="Elokinator")

# Routes
@app.get("/")
def index():
    return FileResponse(str(STATIC / "index.html"))

@app.get("/model.json")
def model_json():
    return FileResponse(str(STATIC / "model.json"))

@app.get("/favicon.ico")
def favicon():
    return FileResponse(str(STATIC / "favicon.ico")) if (STATIC / "favicon.ico").exists() else JSONResponse(status_code=204, content=None)

app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
