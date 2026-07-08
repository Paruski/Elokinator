# Elokinator ♟️

Test de 10 posiciones para estimar tu nivel de ajedrez.

**100% en el navegador** — no requiere servidor. La inferencia del modelo GradientBoosting (300 árboles) se ejecuta en JavaScript puro.

## Cómo usar

Abre `https://paruski.github.io/Elokinator/` (GitHub Pages) o abre `static/index.html` localmente (necesitarás servir `model.json` mediante HTTP).

## Desarrollo

- `static/index.html` — todo el frontend + motor de inferencia JS
- `static/model.json` — modelo exportado (1.45 MB)
- `test_data.json` — datos de las 10 posiciones
- `server.py` — servidor FastAPI (opcional, para desarrollo)
- `best_model.pkl` — modelo Python original (GradientBoosting + StandardScaler)

## Método

El modelo se entrenó con 1635 jugadores reales de Lichess blitz. Para cada jugador se extrajo su CP loss (evaluado con Stockfish 15.1) en las 10 posiciones críticas. El prior de población real es el de Lichess Blitz.

- Modelo: GradientBoostingClassifier (300×7 árboles, max_depth=4)
- Precisión (training): 73% exacto, 86% ±1 banda
- Ajuste Bayesiano con prior poblacional real

## Licencia

MIT
