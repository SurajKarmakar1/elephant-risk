# AGENT.md — Elephant Risk Assessment Tool

## 1. Project Overview

**Goal:** Build a human–elephant conflict (HEC) risk assessment tool from a dataset of
conflict points (`lat`, `lon`, `year`). Output is an explainable, hexagon-based risk map
served through a web dashboard, runnable fully locally (no hosting, no Docker).

**Framing rule:** The model predicts *relative conflict risk*, NOT guaranteed elephant
attacks. All UI text and reports must use "risk score / risk level", never "prediction
of attack". Data is presence-only → pseudo-absence sampling is mandatory, and metrics
must be interpreted as relative ranking quality.

**Non-negotiable stack (do not substitute):**
- Python 3.11, FastAPI + Uvicorn
- SQLite (Python stdlib driver) + SQLAlchemy; geometry stored as GeoJSON text (EPSG:4326)
- GeoPandas, Rasterio, H3 v4
- LightGBM (core model), TabPFN (experimental model)
- SHAP (explainability), MLflow (tracking)
- Next.js + TypeScript + TailwindCSS, MapLibre GL JS, Recharts

**Architecture rationale (for report/evaluators):** All spatial computation happens in
the ML pipeline (GeoPandas/Shapely/H3). The database only stores and serves precomputed
per-cell results, so a spatial database (PostGIS) adds operational cost with no query
benefit. SQLite is the deliberate choice.

---

## 2. Repository Layout

```text
elephant-risk/
├── AGENT.md
├── .env / .gitignore / README.md
├── run_backend.sh / run_frontend.sh
├── data/            # raw/, external/, processed/, predictions/, elephant.db  (gitignored)
├── ml/              # requirements.txt, notebooks/, src/, models/, mlruns/
├── backend/         # requirements.txt, app/{main,config,db}, models/, schemas/, routers/, services/, tests/
└── frontend/        # Next.js app: app/, components/, lib/, types/
```

**Boundary rule:** `ml/` and `backend/` NEVER import from each other. They communicate
only via (a) saved model files in `ml/models/`, (b) GeoJSON in `data/predictions/`,
(c) the SQLite file `data/elephant.db`. Frontend talks only to the FastAPI REST API.

---

## 3. Global Engineering Rules

1. **CRS discipline:** store/serve everything in EPSG:4326. Do ALL distance/buffer math
   in a projected CRS (use the appropriate UTM zone for the study area, set in
   `ml/src/config.py`). Never compute distances in degrees.
2. **H3 grid:** resolution 7 (~5 km² hexagons) as default, configurable in `config.py`.
   One hexagon = one risk unit. Cell ID = H3 index string, used as key everywhere.
   Use H3 v4 API names (`latlng_to_cell`, `cell_to_boundary`, `polygon_to_cells`).
3. **No data leakage:** features derived from conflict history (e.g., past-conflict
   density) must use ONLY events strictly before the prediction year. Spatial block
   splits for evaluation; random splits are forbidden.
4. **No spatial SQL:** SQLite has no spatial index. Never filter by geometry in SQL.
   Point→cell lookups use `h3.latlng_to_cell()` in Python; DB queries filter only on
   indexed scalar columns (year, risk_level, cell_id).
5. **Reproducibility:** every training run logs params, metrics, git commit hash, and
   feature list to MLflow. Fixed random seeds in `config.py`.
6. **Small steps:** each phase below must end in a runnable, verifiable state before
   moving on. Do not build later phases early.
7. **Secrets/paths:** DB path and settings live in `.env`, never hardcoded. `data/` and
   `mlruns/` are gitignored.
8. **Typing:** type hints in all Python; strict TypeScript in frontend. Pydantic schemas
   for every API request/response.

---

## 4. Data Contracts

### Input CSV (`data/raw/conflict_points.csv`)
Required columns: `lat` (float), `lon` (float), `year` (int).
Optional passthrough: `date`, `type`, `village`, `district`.
Ingest must: drop invalid coords/year, deduplicate, report row counts before/after,
and flag points outside the study-area bounding box (bbox defined in `config.py`).

### Feature table (`data/processed/features.parquet`)
One row per (h3_cell, year). Minimum v1 features (OSM-derived is acceptable):
- `dist_forest_km`, `dist_road_km`, `dist_water_km`, `dist_settlement_km`
- `elev_m`, `slope_deg` (from DEM raster)
- `past_conflict_density` (events within 10 km, years < current year)
- `year`
Later additions: NDVI, rainfall, cropland proximity, population density.

### Labels
- Positive: cell-year containing ≥1 conflict event.
- Negative: pseudo-absence cells sampled OUTSIDE a 5 km buffer of any known conflict
  point, matched count to positives, seed fixed. Documented in `labels.py` docstring.

### Risk output (`data/predictions/risk_{year}.geojson`)
FeatureCollection of hexagon polygons, properties:
```json
{ "cell_id": "...", "year": 2026, "risk_score": 0.82,
  "risk_level": "high", "top_factors": ["dist_forest_km", "past_conflict_density", "dist_water_km"] }
```
Risk levels from fixed thresholds defined ONCE in `ml/src/config.py` and mirrored in
`frontend/src/lib/riskColors.ts`:
low < 0.25 ≤ moderate < 0.5 ≤ high < 0.75 ≤ critical.

### Database (`data/elephant.db`, plain SQLite via SQLAlchemy)
```text
conflict_events(id PK, lat, lon, year, attrs_json TEXT)
risk_cells(cell_id, year, model_version, geom_geojson TEXT,
           risk_score REAL, risk_level TEXT, top_factors_json TEXT)
    -- PK(cell_id, year, model_version); INDEX(year, risk_level)
model_versions(id PK, name, trained_at, metrics_json TEXT, artifact_path)
```
All geometry is GeoJSON text in EPSG:4326. `/risk-map` reads rows and wraps
`geom_geojson` into a FeatureCollection — no spatial operations in the DB layer.

---

## 5. API Contract (build exactly these first)

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | `{status: "ok"}` |
| GET | `/risk-map?year=&level=` | GeoJSON FeatureCollection of risk cells |
| POST | `/predict` `{lat, lon, year}` | `{cell_id, risk_score, risk_level, top_factors[]}` |
| GET | `/explain/{cell_id}?year=` | SHAP factor list `[{feature, value, contribution}]` |
| GET | `/conflicts?year_from=&year_to=` | GeoJSON of historical conflict points |
| GET | `/hotspots?year=&limit=` | Top-N cells sorted by risk_score |

CORS: allow `http://localhost:3000`. Errors: JSON `{detail}` with proper status codes.
`/predict` resolves the containing cell via `h3.latlng_to_cell(lat, lon, RES)` in pure
Python, then looks up the precomputed row; if absent, computes features on the fly
using a small pure feature-builder function duplicated into backend services (do NOT
cross-import the ml package).

---

## 6. Phased Plan

### Phase 0 — Scaffold & EDA
Tasks: create folder tree, `.env`, `.gitignore`, both `requirements.txt`,
`ml/src/config.py` (paths incl. `data/elephant.db`, bbox, UTM zone, H3 res, seed,
thresholds); `notebooks/01_eda.ipynb`: load CSV, clean, plot points by year, summary stats.
**Done when:** notebook runs end-to-end; cleaned point count and year range documented
in README.

### Phase 1 — Grid & Features
Tasks: `ingest.py`, `grid.py` (H3 polyfill of study bbox/region), `features.py`
(download/prepare OSM layers + DEM into `data/external/`, compute distance features in
UTM, write `features.parquet`), `labels.py`, `split.py` (spatial blocks via coarse H3
res-4 grouping + holdout-final-year temporal split).
**Done when:** `features.parquet` exists with labels, no NaNs in required columns,
and a quick map plot of the grid colored by label looks sane.

### Phase 2 — Baseline Model
Tasks: `train_lgbm.py` (MLflow logging, early stopping on spatial validation),
`evaluate.py` (PR-AUC, ROC-AUC, recall@top-10% cells, calibration curve; report per
held-out year), save best model to `ml/models/lgbm_latest.txt` + metrics JSON.
**Done when:** one command trains + evaluates + saves; metrics logged in MLflow;
README records baseline numbers.

### Phase 3 — Prediction Pipeline
Tasks: `predict.py` (score full grid for each year → GeoJSON + upsert into
`risk_cells` in SQLite; register row in `model_versions`; create tables/indexes if
missing), `explain.py` (SHAP TreeExplainer, per-cell top-3 factors persisted).
**Done when:** GeoJSON files exist per year, DB rows loaded, spot-check of 3 known
conflict locations shows elevated risk.

### Phase 4 — Backend API
Tasks: FastAPI app per Section 5, SQLAlchemy models matching Section 4 schema, DB
session (read-only usage; `check_same_thread=False`), inference service (loads LightGBM
once at startup), geo service (read rows → build GeoJSON), pytest for every endpoint
incl. invalid lat/lon and unknown cell_id.
**Done when:** `uvicorn app.main:app --reload` serves all endpoints; tests pass;
Swagger docs at `/docs` show correct schemas.

### Phase 5 — Frontend Dashboard
Tasks: Next.js scaffold; pages: `/` (stats cards + links), `/map` (MapLibre base map +
H3 fill layer colored by risk_level, conflict point toggle, year filter, click hex →
CellDetailPanel showing score + top factors from `/explain`), `/hotspots` (table from
`/hotspots`), `/insights` (placeholder until Phase 6).
**Done when:** map renders real predictions, clicking a hexagon shows its explanation,
year filter re-fetches correctly, no console errors.

### Phase 6 — TabPFN Comparison & Insights
Tasks: `train_tabpfn.py` (same features/splits; subsample deterministically if size
exceeds TabPFN limits — log the cap), compare vs LightGBM in `evaluate.py`, comparison
chart data exposed via `/insights` endpoint or static JSON; frontend `/insights` page
with Recharts bar charts (PR-AUC, recall@top-k per model) + global SHAP summary.
**Done when:** comparison table in README with honest verdict; winner declared the
production model; insights page renders.

### Phase 7 — Polish & Report Assets
Tasks: README (setup, run instructions, architecture diagram, results), export
screenshots/plots for the report, `run_backend.sh` / `run_frontend.sh`, final cleanup
(remove dead code, lint).
**Done when:** a fresh clone + README instructions reproduces the full demo on a
second machine without asking the author anything (no DB server to install — the
SQLite file is regenerated by `predict.py`).

---

## 7. Verification Checklist (run before claiming any phase done)

- [ ] Code runs from a clean shell with only the documented commands
- [ ] No cross-imports between ml/ and backend/
- [ ] All geometry in EPSG:4326; all distances computed in UTM
- [ ] No geometry filtering in SQL; cell lookups via H3 in Python only
- [ ] No future-year information in any feature (leakage audit)
- [ ] Evaluation uses spatial blocks AND a held-out year — never random split
- [ ] Every metric quoted in README traces to an MLflow run
- [ ] UI wording says "risk", never "attack prediction"

## 8. Known Pitfalls (read before coding)

- Pseudo-absences too close to positives → inflated metrics. Keep the 5 km buffer.
- TabPFN input-size limits → subsample deterministically, log the subsample size.
- H3 v4 API names differ from v3 (`latlng_to_cell`, `cell_to_boundary`,
  `polygon_to_cells`); pin the version in requirements and use v4 names consistently.
- MapLibre needs polygon GeoJSON, not H3 indexes — convert via `cell_to_boundary`
  and close the ring (first coordinate repeated at the end).
- OSM downloads can be huge — clip to study bbox FIRST, cache in `data/external/`.
- SQLite: enable WAL mode if backend reads while predict.py writes; keep writes in
  the ml pipeline only. Never store WKB/Shapely objects — GeoJSON text only.
- Windows + GeoPandas install issues → use conda-forge or prebuilt wheels; document
  whichever works in README.
