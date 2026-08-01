## PriceWise C2B Auto Pricing Product

PriceWise is an internal-style pricing decision product for consumer-to-business
vehicle acquisition. A pricing user enters vehicle details, and the system
predicts market value, recommends an acquisition offer, estimates margin, flags
risk, and records whether the recommendation was accepted, rejected, or
overridden.

This public project uses a used-car dataset and contains no proprietary company
data or code.

## Business Problem

In C2B vehicle acquisition, pricing teams need to answer:

- What is the expected market value of this vehicle?
- What offer should we make to the customer?
- What margin can we expect after reconditioning cost?
- Does this quote need manager approval?
- Which pricing recommendations were accepted, rejected, or overridden?

## Current Architecture

```text
used_cars.csv
    -> data ingestion
    -> train/test split
    -> feature engineering and preprocessing
    -> CatBoost regression model
    -> saved model and preprocessor artifacts
    -> FastAPI pricing API
    -> decision logging database
    -> React pricing console
```

## Project Flow

1. The raw dataset is loaded from `notebook/data/used_cars.csv`.
2. The ingestion component creates train and test datasets under `artifact/`.
3. The transformation component cleans mileage and price fields, creates vehicle
   age and interaction features, encodes categorical columns, and scales numeric
   columns.
4. The model trainer compares regression models and saves the selected model to
   `artifactS/model.pkl`.
5. The prediction pipeline loads the model and preprocessor artifacts and
   returns predicted log price.
6. The pricing service converts predicted log price to market value, applies
   business pricing rules, and returns an operational recommendation.
7. The FastAPI backend exposes market price, recommendation, decision logging,
   and model health endpoints.

## API Endpoints

```text
GET  /api/v1/health
POST /api/v1/market-price
POST /api/v1/recommendations
POST /api/v1/pricing-decisions
GET  /api/v1/pricing-decisions
GET  /api/v1/model-health
```

Example recommendation request:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/recommendations \
  -H "Content-Type: application/json" \
  -d @examples/recommendation_request.json
```

Example decision logging request:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/pricing-decisions \
  -H "Content-Type: application/json" \
  -d @examples/decision_request.json
```

Example product response:

```json
{
  "market_value": 17268.48,
  "recommended_offer": 12900,
  "expected_margin_pct": 0.1835,
  "risk_level": "High",
  "approval_required": true,
  "recommended_action": "Send to manager review before making offer."
}
```

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI backend:

```bash
uvicorn api.main:app --reload
```

Run the React frontend:

```bash
cd frontend
pnpm install
pnpm dev
```

Optional legacy Flask app:

```bash
python application.py
```

Run tests:

```bash
python -m unittest tests/test_pricing_recommendation.py
```

## Production Extension

The current implementation demonstrates the core pricing workflow using a local
dataset, saved model artifacts, FastAPI services, and SQLite decision logging.
In a production environment, the same design can be extended with batch feature
pipelines on S3, SageMaker training and model registry, an ECS-hosted FastAPI
service, RDS/PostgreSQL for decision history, CloudWatch monitoring, and a
React frontend used by pricing or sales operations teams.

## Portfolio Context

This project is a public portfolio implementation of an auto-pricing workflow.
It is designed to demonstrate machine learning model serving, pricing decision
logic, API development, decision tracking, and production-ready architecture
patterns without using proprietary company data or code.
