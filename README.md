# Egypt Food Import Intelligence DSS

Streamlit decision-support prototype for an Intelligent Decision Support Systems (IDSS) course. It fuses **WFP** retail prices with **optional FAO / FAOSTAT tracks** (wheat spine + multi-commodity catalog for all WFP goods when you provide FAOSTAT), **Arabic–English NLP** on news headlines, **shipping / port** telemetry, **GDELT** conflict aggregates, and **daily port activity** into forecasting, risk scoring, GIS-style overlays, goal programming, AHP weighting, discriminant risk classification, and a rule-based inference layer.

## Data (training & features)

Place these CSV files in the project root (expected defaults):

- `wfp_food_prices_egy.csv` — WFP Egypt retail series; **forecasting** and baseline price stress.
- `fao_wheat_egy.csv` — optional **single wheat-focused** FAOSTAT (or simple monthly `month,price_usd`) export; merges into the **pipeline wheat spine** (55%/45% FAO/WFP weights when overlapping months exist — see `merge_wfp_fao`).
- `fao_commodities_egy.csv` — optional **multi-item** FAOSTAT export for Egypt (`Area`, `Item`, `Value`, `Year`, …). Each **Item** becomes a monthly series; the app picks the closest Item name match for **every commodity** listed in `wfp_food_prices_egy.csv` and blends USD series the same way. You can ship one combined FAOSTAT download here instead of per-commodity files.
- `gdelt_conflict_1_0.csv` — conflict/event intensity by country; geopolitical layer.
- `Daily_Port_Activity_Data_and_Trade_Estimates.csv` — chunked read; **logistics / shipping proxy** for Egypt + corridors.
- `shipping_metrics.csv` — optional **shipping tracker** KPIs (congestion, delays, berth use), forward-filled to monthly.
- `news_headlines_bilingual.csv` — `published_at`, `text`, `language` (`ar` / `en`); drives **NLP conflict + sentiment** indices. If missing, the pipeline synthesizes demo headlines from price spikes (clearly flagged in the UI).

Optional: set **`SHIPPING_TRACKER_API_URL`** to a JSON endpoint for live vessel/congestion APIs.

**Demo maritime CSVs** (offline, no AIS API) in `data/sample/`: `demo_ports.csv`, `demo_vessels.csv`, `demo_conflict_zones.csv`, `demo_routes.csv`. If missing, the app falls back to the same data from in-code defaults.

The discriminant-analysis **labels** remain rule-generated for educational use (see `PROJECT_REPORT.md`).

## Modules (P-02 tech stack)

| Layer | Files |
|--------|--------|
| Time series / blend | `src/data_pipeline.py`, `src/fao_prices.py`, `src/decision_methods.py` |
| Arabic–English NLP | `src/nlp_conflict_index.py` |
| Shipping | `src/shipping_tracker.py` |
| Risk dashboard | `app.py`, `src/ui_components.py` |
| Port / vessel intelligence (synthetic AIS demo) | `src/port_intelligence.py`, `src/vessel_tracking.py`, `src/conflict_zones.py`, `src/route_risk.py`, `src/maritime_viz.py`, `src/ui_maritime_dashboard.py`, `src/alert_system.py`, `src/ollama_advisor.py`, `data/sample/*.csv` |

## Run

```bash
cd IDSS_P
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows, if your default `python` is MSYS/MinGW and wheels fail to install, use **Python 3.10+ from python.org** (or the `py -3.12` launcher) so NumPy/Pandas install from wheels.

## IDSS concepts mapped in the app

| Concept | Where it appears |
|--------|------------------|
| Data management | `src/data_pipeline.py`, safe loaders |
| Model management | `src/decision_methods.py`, `src/goal_programming.py`, `src/ahp.py`, `src/discriminant_analysis.py`, `src/gis_analysis.py`, `src/fao_prices.py`, `src/shipping_tracker.py`, `src/nlp_conflict_index.py` |
| Knowledge base | `src/knowledge_base.py` |
| Inference engine | `src/inference_engine.py` |
| Explanation facility | `src/explanation_engine.py`, recommendation cards |
| User interface | `app.py`, `src/ui_components.py` |

## Project report

See `PROJECT_REPORT.md` for methodology write-up and architecture diagrams.
