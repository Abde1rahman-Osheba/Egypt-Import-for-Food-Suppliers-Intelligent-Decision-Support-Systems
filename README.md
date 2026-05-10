# 🌾 Egypt Food Import Intelligence DSS  
### From scattered global signals to early warnings for Egypt’s food import supply chain

A **Streamlit-based Intelligent Decision Support System (IDSS)** prototype built for analyzing, forecasting, and explaining risks in Egypt’s food import ecosystem.

This project turns raw signals from **food prices, FAOSTAT commodity tracks, bilingual Arabic–English news, geopolitical conflict data, port activity, shipping indicators, and synthetic maritime intelligence** into a decision-support dashboard for early disruption awareness.

Instead of waiting for shortages, price spikes, or shipment delays to become obvious, this DSS helps stakeholders ask:

> **“Is a food import disruption forming — and what should we do before it becomes a crisis?”**

---

## 🚀 Project Vision

Egypt relies heavily on imported strategic commodities, especially wheat and other food staples. Global disruptions such as conflict, export restrictions, port congestion, freight delays, or sudden price movements can quickly affect local food availability and import costs.

The goal of this system is to support **proactive decision-making** by combining multiple weak signals into one explainable risk view.

The system is designed as an educational but realistic IDSS prototype that demonstrates:

- 📈 Forecasting food price stress
- 🌍 Monitoring geopolitical and conflict exposure
- ⚓ Tracking logistics and port activity indicators
- 📰 Extracting risk signals from Arabic and English news headlines
- 🧠 Applying IDSS methods such as AHP, goal programming, discriminant analysis, inference rules, and explanation facilities
- 🗺️ Visualizing ports, vessels, conflict zones, and route risks in a GIS-style dashboard

---

## ✨ What Makes This Project Different?

Most dashboards show data.  
This system tries to **reason over data**.

It does not only answer:

> “What happened?”

It also supports:

> “Why is this risky?”  
> “Which commodity is under pressure?”  
> “Which routes or ports may be affected?”  
> “What is the recommended action?”  
> “How did the system reach this decision?”

That makes it more than a visualization tool — it is a decision-support prototype with a knowledge base, inference layer, explainable recommendations, and multiple decision-analysis methods.

---

## 🧩 Main Features

### 1. Food Price Forecasting & Stress Detection

The app uses WFP Egypt retail food price data as the main forecasting source.

It can also blend WFP prices with optional FAO / FAOSTAT commodity data to create a stronger commodity-level price spine.

Supported price intelligence includes:

- Monthly food price trend analysis
- Forecasting baseline price movement
- Price stress indicators
- WFP + FAO blended commodity signals
- Wheat-focused analysis
- Multi-commodity catalog matching

---

### 2. FAOSTAT Commodity Blending

The system supports two FAO data modes:

| File | Purpose |
|---|---|
| `fao_wheat_egy.csv` | Optional wheat-focused FAOSTAT file |
| `fao_commodities_egy.csv` | Optional multi-item FAOSTAT export for Egypt |

When FAO and WFP overlap, the system blends them using a weighted approach:

```text
55% FAO + 45% WFP
```

This gives the pipeline a stronger commodity intelligence layer while keeping WFP retail prices as the main local signal.

---

### 3. Arabic–English News Risk Intelligence

The NLP layer processes bilingual news headlines to detect possible food import disruption signals.

Supported headline inputs:

```text
news_headlines_bilingual.csv
```

Expected columns:

```text
published_at, text, language
```

Supported languages:

```text
ar, en
```

The NLP module extracts signals related to:

- Conflict
- Supply disruption
- Price pressure
- Food security
- Shipping and logistics risk
- Negative or alarming sentiment

If the news file is missing, the app generates demo headlines from price spikes and clearly marks them as synthetic in the UI.

---

### 4. Geopolitical Conflict Layer

The system uses GDELT-style conflict aggregates to model international disruption exposure.

Expected file:

```text
gdelt_conflict_1_0.csv
```

This layer helps connect external events to import risk, especially when conflict appears near:

- Exporting countries
- Shipping corridors
- Maritime chokepoints
- Regional trade routes
- Strategic food supply areas

---

### 5. Shipping, Port, and Maritime Intelligence

The dashboard includes a synthetic AIS-style maritime intelligence layer for offline demonstrations.

It can visualize:

- Egyptian and regional ports
- Vessel positions
- Port activity
- Route paths
- Conflict zones
- Risk overlays
- Alternative route suggestions
- Vessel and port information cards

Demo maritime files are located in:

```text
data/sample/
```

Expected demo files:

```text
demo_ports.csv
demo_vessels.csv
demo_conflict_zones.csv
demo_routes.csv
```

If these files are missing, the app falls back to in-code demo defaults.

---

### 6. Route Risk and Rerouting Support

The maritime layer supports route-level risk reasoning.

For each vessel or route, the system can evaluate:

- Whether the route passes near a risk zone
- Whether conflict zones may delay movement
- Whether an alternative route should be suggested
- Why a route may be safer or riskier

The route advisor can optionally connect to Ollama for local LLM-based explanation and recommendation support.

---

### 7. Decision-Support Methods

This project maps directly to Intelligent Decision Support Systems concepts.

| IDSS Concept | Implementation |
|---|---|
| Data Management | Safe loaders and preprocessing pipelines |
| Model Management | Forecasting, risk scoring, AHP, goal programming, discriminant analysis |
| Knowledge Base | Structured rules and domain knowledge |
| Inference Engine | Rule-based reasoning over risk indicators |
| Explanation Facility | Recommendation cards and reasoning summaries |
| User Interface | Streamlit dashboard and maritime visual interface |

---

## 🧠 IDSS Methods Used

The project demonstrates several decision-support techniques.

### AHP Weighting

Used to prioritize decision criteria such as:

- Price stress
- Conflict intensity
- Port congestion
- Shipping delay
- News risk
- Commodity importance

### Goal Programming

Used to model trade-offs between competing goals, such as:

- Minimize import risk
- Minimize expected delay
- Minimize cost pressure
- Maintain supply stability

### Discriminant Risk Classification

Used to classify risk states based on generated educational labels.

> Note: The discriminant-analysis labels are rule-generated for course demonstration purposes. See `PROJECT_REPORT.md` for methodology details.

### Rule-Based Inference

The inference engine converts signals into explainable alerts, such as:

- High price stress
- High geopolitical exposure
- Possible port disruption
- Elevated commodity risk
- Route danger warning
- Suggested monitoring or rerouting action

---

## 🗂️ Data Files

Place the following CSV files in the project root.

| File | Required | Purpose |
|---|---:|---|
| `wfp_food_prices_egy.csv` | Yes | Main WFP Egypt retail food price series |
| `fao_wheat_egy.csv` | Optional | Wheat-focused FAOSTAT or monthly wheat USD series |
| `fao_commodities_egy.csv` | Optional | Multi-commodity FAOSTAT export for Egypt |
| `gdelt_conflict_1_0.csv` | Recommended | Geopolitical conflict/event intensity |
| `Daily_Port_Activity_Data_and_Trade_Estimates.csv` | Recommended | Daily port activity and logistics proxy |
| `shipping_metrics.csv` | Optional | Congestion, delay, berth use, and shipping KPIs |
| `news_headlines_bilingual.csv` | Optional | Arabic–English news headlines for NLP risk index |

---

## ⚓ Demo Maritime Data

Offline maritime demo files are stored in:

```text
data/sample/
```

| File | Description |
|---|---|
| `demo_ports.csv` | Port locations and metadata |
| `demo_vessels.csv` | Synthetic vessel positions and voyage information |
| `demo_conflict_zones.csv` | Risk zones, conflicts, or disruption areas |
| `demo_routes.csv` | Vessel routes and corridor paths |

These files allow the maritime dashboard to work without a live AIS API.

---

## 🌐 Optional Live Shipping API

You can connect a live shipping or vessel-tracking API by setting:

```bash
SHIPPING_TRACKER_API_URL="your_json_endpoint_here"
```

The system will use the endpoint as a live telemetry source when available.

---

## 🏗️ Project Structure

```text
IDSS_P/
│
├── app.py
├── requirements.txt
├── PROJECT_REPORT.md
│
├── src/
│   ├── data_pipeline.py
│   ├── fao_prices.py
│   ├── decision_methods.py
│   ├── goal_programming.py
│   ├── ahp.py
│   ├── discriminant_analysis.py
│   ├── gis_analysis.py
│   ├── shipping_tracker.py
│   ├── nlp_conflict_index.py
│   ├── knowledge_base.py
│   ├── inference_engine.py
│   ├── explanation_engine.py
│   ├── ui_components.py
│   │
│   ├── port_intelligence.py
│   ├── vessel_tracking.py
│   ├── conflict_zones.py
│   ├── route_risk.py
│   ├── maritime_viz.py
│   ├── ui_maritime_dashboard.py
│   ├── alert_system.py
│   └── ollama_advisor.py
│
└── data/
    └── sample/
        ├── demo_ports.csv
        ├── demo_vessels.csv
        ├── demo_conflict_zones.csv
        └── demo_routes.csv
```

---

## 🧪 Tech Stack

| Layer | Tools |
|---|---|
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Forecasting & Analysis | Python time-series and statistical methods |
| NLP | Arabic–English headline processing |
| Decision Methods | AHP, goal programming, discriminant analysis |
| GIS / Maritime View | Map-based overlays and route visualization |
| LLM Advisor | Optional Ollama integration |
| Reporting | `PROJECT_REPORT.md` |

---

## ▶️ How to Run

Clone the repository:

```bash
git clone https://github.com/Abde1rahman-Osheba/Egypt-Import-for-Food-Suppliers-Intelligent-Decision-Support-Systems.git
cd Egypt-Import-for-Food-Suppliers-Intelligent-Decision-Support-Systems
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 🪟 Windows Notes

If package installation fails on Windows because your default `python` points to MSYS/MinGW, use Python 3.10+ from python.org.

You can also try:

```bash
py -3.12 -m pip install -r requirements.txt
py -3.12 -m streamlit run app.py
```

---

## 📊 Dashboard Capabilities

The Streamlit dashboard includes sections for:

- Food price trends
- Forecasting and price stress
- FAO/WFP blended commodity analysis
- Geopolitical conflict risk
- News-based NLP risk index
- Shipping and logistics indicators
- Port and vessel intelligence
- Route risk and conflict-zone overlays
- IDSS recommendations
- Explanation cards
- Rule-based alerts

---

## 🧭 Example Decision Questions

This DSS helps answer questions such as:

- Which commodities are showing early price stress?
- Are geopolitical events increasing import risk?
- Are port or shipping indicators showing possible delay?
- Which vessels or routes are close to conflict zones?
- Should a route be monitored, rerouted, or kept as-is?
- Which risk factor contributes most to the final score?
- What action should a decision-maker consider next?

---

## 📌 Educational Scope

This project is designed for an **Intelligent Decision Support Systems course**.

It focuses on demonstrating how DSS components can work together:

```text
Data → Models → Knowledge Base → Inference → Explanation → Decision Support
```

Some data layers may be synthetic, optional, or rule-generated for academic demonstration. The system is not intended as a production-grade national food security platform without further validation, live data integration, and expert review.

---

## 📄 Project Report

For methodology, architecture diagrams, decision-support design, and educational explanation, see:

```text
PROJECT_REPORT.md
```

---

## 🧠 Core Idea

Food import disruption rarely starts as one obvious signal.

It begins as a pattern:

- A price starts rising
- A port slows down
- A headline becomes more alarming
- A conflict zone expands
- A route becomes riskier
- A commodity becomes harder to secure

This project connects those signals into one intelligent dashboard.

> **The aim is not only to predict disruption — but to explain it, visualize it, and support better decisions before it becomes visible in the market.**


## 🔗 Repository

GitHub Repository:

```text
https://github.com/Abde1rahman-Osheba/Egypt-Import-for-Food-Suppliers-Intelligent-Decision-Support-Systems
```

If this project helped you understand how intelligent decision-support systems can combine forecasting, NLP, GIS, shipping intelligence, and rule-based reasoning, consider starring the repository.
