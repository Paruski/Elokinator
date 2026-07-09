# Elokinator ♟️

Test de 10 posiciones para estimar tu nivel de ajedrez.

**100% en el navegador** — no requiere servidor. La inferencia del modelo RandomForest (400 árboles) se ejecuta en JavaScript puro.

## Cómo usar

Abre `https://paruski.github.io/Elokinator/` (GitHub Pages) o abre `static/index.html` sirviendo el directorio con:

```bash
cd static && python3 -m http.server 8080
```

## Método

### Datos
- **2418 jugadores reales** de Lichess blitz (extraídos de 75760 partidas etiquetadas)
- 7 bandas de nivel basadas en la población real de Lichess Blitz (μ≈1487, σ≈395)
- Para cada jugador se extrajo su CP loss (evaluado con Stockfish 15.1 a profundidad 14) en las 10 posiciones más discriminantes

### Modelo
- **RandomForestClassifier**: 400 árboles, max_depth=10, min_samples_leaf=5
- Características: 10 raw CP losses (uno por posición)
- Precisión (5-fold CV): 32.3% exacto, 68.2% ±1 banda, MAE=1.25
- Prior poblacional real de Lichess Blitz incorporado en la inferencia

### Las 10 posiciones
Seleccionadas mediante análisis bifásico de 36 posiciones candidatas:
1. Cribado inicial por V de Cramér (asociación jugada × banda)
2. Evaluación con Stockfish del centipawn loss de cada jugada
3. Score combinado: 0.5·ANOVA-F_z + 0.25·CramerV_z + 0.25·|Spearman-ρ|_z

## Estructura

```
static/
├── index.html     — Frontend completo + motor de inferencia JS
└── model.json     — RandomForest exportado (7.3 MB, 400 árboles)
```

## Licencia

MIT
