# Inventory Transfer Optimization

A Streamlit application that forecasts retail demand, identifies inventory
imbalances, and recommends cost-aware product transfers between stores.

![Inventory network illustration](assets/images/dashboard_hero.png)

## What the application does

The application combines demand forecasting and inventory optimization in one
operational workflow:

```text
Sample data or uploaded ZIP
            |
            v
Validate and load seven CSV files
            |
            v
Forecast demand for every store-product pair
            |
            v
Calculate shortage, balanced, and excess inventory
            |
            v
Generate a transfer plan with the selected optimizer
            |
            v
Review KPIs, charts, tables, and routes on the dashboard
```

## Main features

- Built-in sample dataset with 365 days of sales history
- ZIP upload for operational datasets
- Four demand-forecasting methods:
  - Historical Average
  - Moving Average
  - Random Forest
  - AdaBoost
- Three inventory-transfer optimizers:
  - Greedy
  - Linear Programming
  - Genetic Algorithm
- Dataset-specific model training and reusable model artifacts
- Interactive forecast, inventory, and transfer filters
- Inventory and transfer KPIs
- Downloadable transfer plans and source tables
- OpenStreetMap network map with store markers and transfer routes
- No map API key required

## Replenishment policy

The user can request a planning horizon from 1 to 14 days. The optimization
pipeline converts that request into one of two replenishment targets:

| Requested horizon | Inventory replenishment target |
| --- | --- |
| 1-7 days | Replenish enough inventory for 7 days |
| 8-14 days | Replenish enough inventory for 14 days |

The requested horizon is still used for demand forecasting. The replenishment
target determines the stock level used by the inventory analyzer.

## Dashboard pages

| Page | Purpose |
| --- | --- |
| Overview | Summarizes shortage recovery, transfer volume, cost, and run settings |
| Demand Forecast | Explores predicted demand by city, store, category, and product |
| Inventory Health | Shows shortage, balanced, and excess store-product records |
| Transfer Plan | Reviews planned routes, quantities, costs, distance, and lead time |
| Network Map | Displays store health and the highest-volume transfer routes |
| Data & Models | Inspects source data, trains ML models, and shows feature importance |

## Quick start

### Windows PowerShell

Create the virtual environment:

```powershell
py -m venv .venv
```

Install the dependencies without needing to activate PowerShell scripts:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start the application:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### macOS or Linux

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m streamlit run app.py
```

Streamlit normally opens the application at `http://localhost:8501`.

## Using the application

1. Select **Sample Dataset** or **Upload ZIP Dataset** in the sidebar.
2. Choose a planning horizon from 1 to 14 days.
3. Select a forecast method and an optimization algorithm.
4. If Random Forest or AdaBoost has no compatible artifact, open
   **Data & Models** and train the selected model first.
5. Select **Run optimization**.
6. Explore the dashboard pages and download the required results.

Model artifacts are associated with a fingerprint of the active dataset. A
model trained for a different dataset is not used accidentally.

## ZIP dataset format

An uploaded ZIP must contain these seven CSV files at its root:

```text
stores.csv
products.csv
sales_data.csv
inventory_data.csv
distance_matrix.csv
duration_matrix.csv
transport_cost_matrix.csv
```

Required columns for the main business tables:

| File | Required columns |
| --- | --- |
| `stores.csv` | `store_id`, `store_name`, `city`, `latitude`, `longitude` |
| `products.csv` | `product_id`, `product_name`, `category`, `cost`, `price` |
| `sales_data.csv` | `date`, `store_id`, `product_id`, `quantity_sold`, `revenue`, `cost_of_goods_sold` |
| `inventory_data.csv` | `store_id`, `product_id`, `current_stock`, `last_updated` |

The three matrix files use `store_id` as the first column and one column for
each destination store ID.

## Generate sample data

To regenerate all sample CSV files:

```powershell
.\.venv\Scripts\python.exe -m src.data_generator.generate_all
```

This replaces generated files in `data/`, so review the Git diff before
committing regenerated data.

## Project structure

```text
inventory-transfer-optimization/
|-- app.py                    # Streamlit application entry point
|-- requirements.txt          # Python dependencies
|-- README.md                 # Project documentation
|-- assets/                   # Dashboard images and CSS
|-- data/                     # Sample CSV datasets
|-- models/                   # Locally trained model artifacts
|-- results/                  # Generated optimization results
|-- src/
|   |-- config.py             # Shared project settings
|   |-- data_generator/       # Sample retail data generation
|   |-- data_ingestion/       # ZIP upload and extraction
|   |-- dashboard/            # Streamlit UI and dashboard services
|   |-- forecasting/          # Demand forecasting models
|   |-- optimizers/           # Inventory transfer algorithms
|   |-- data_loader.py        # Load CSV files into ProjectData
|   |-- validator.py          # Validate input datasets
|   |-- inventory_analyzer.py # Detect shortage and excess inventory
|   |-- route_analyzer.py     # Analyze transfer routes
|   |-- optimization_pipeline.py
|   `-- metrics.py            # Evaluate transfer-plan performance
`-- tests/                    # Automated tests
```

### Component responsibilities

| Component | Responsibility |
| --- | --- |
| `app.py` | Starts the Streamlit application and coordinates the dashboard workflow. |
| `src/config.py` | Stores file paths, forecasting settings, inventory policies, routing limits, and optimizer settings. |
| `src/data_generator/` | Generates stores, products, sales history, inventory, distance, duration, and transport-cost data. |
| `src/data_ingestion/` | Reads an uploaded ZIP file and verifies that it contains the seven required CSV files. |
| `src/dashboard/` | Contains reusable UI components, charts, session state, model management, map construction, and dashboard pages. |
| `src/forecasting/` | Prepares time-series features and forecasts demand using statistical and machine-learning methods. |
| `src/inventory_analyzer.py` | Compares current stock with forecast demand and classifies each store-product pair as shortage, balanced, or excess. |
| `src/route_analyzer.py` | Combines distance, travel time, and transport cost information for possible transfer routes. |
| `src/optimizers/` | Creates transfer plans using Greedy, Linear Programming, or Genetic Algorithm methods. |
| `src/optimization_pipeline.py` | Connects forecasting, inventory analysis, routing, optimization, and evaluation into one workflow. |
| `src/metrics.py` | Calculates shortage recovery, transfer volume, transport cost, and other optimization KPIs. |
| `data/` | Contains the sample dataset used by the application. |
| `models/` | Stores dataset-specific Random Forest and AdaBoost artifacts locally. |
| `tests/` | Verifies forecasting, optimization, data ingestion, and supporting business logic. |

## Run checks

Run the complete test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Check Python syntax and imports used by the dashboard:

```powershell
.\.venv\Scripts\python.exe -m compileall -q app.py src\dashboard
```

## Planned enhancements

- Optuna hyperparameter tuning
- SHAP explanations for trained forecasting models
- Model comparison and experiment tracking
- Authentication and role-based operational access
- Deployment configuration and production monitoring
