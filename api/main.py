from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.pricing.decision_store import DecisionStore
from src.pricing.recommendation_service import PricingRecommendationService


app = FastAPI(
    title="PriceWise C2B Pricing API",
    version="1.0.0",
    description="Internal pricing decision API for vehicle acquisition offers.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pricing_service = PricingRecommendationService()
decision_store = DecisionStore()


class VehicleRequest(BaseModel):
    brand: str = Field(..., example="Ford")
    model: str = Field(..., example="Utility Police Interceptor Base")
    model_year: int = Field(..., example=2013)
    milage: str = Field(..., example="51,000 mi.")
    fuel_type: str = Field(..., example="E85 Flex Fuel")
    engine: str = Field(..., example="300.0HP 3.7L V6 Cylinder Engine Flex Fuel Capability")
    transmission: str = Field(..., example="6-Speed A/T")
    ext_col: str = Field(..., example="Black")
    int_col: str = Field(..., example="Black")
    accident: str = Field(..., example="At least 1 accident or damage reported")
    clean_title: str = Field(..., example="Yes")


class PricingContext(BaseModel):
    reconditioning_cost: float = Field(1200.0, ge=0)
    target_margin_pct: float = Field(0.12, ge=0, le=0.5)


class RecommendationRequest(BaseModel):
    vehicle: VehicleRequest
    pricing_context: PricingContext = PricingContext()


class DecisionRequest(BaseModel):
    quote_id: str
    user_id: str = Field(..., example="pricing_user_01")
    decision: str = Field(..., example="accepted")
    recommended_offer: float
    final_offer: Optional[float] = None
    reason: Optional[str] = None


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "service": "pricewise-c2b-pricing-api"}


@app.post("/api/v1/market-price")
def predict_market_price(vehicle: VehicleRequest):
    market_value = pricing_service.predict_market_value(vehicle.dict())
    return {"market_value": round(market_value, 2)}


@app.post("/api/v1/recommendations")
def create_recommendation(request: RecommendationRequest):
    return pricing_service.recommend_offer(
        request.vehicle.dict(),
        request.pricing_context.dict(),
    )


@app.post("/api/v1/pricing-decisions")
def record_pricing_decision(request: DecisionRequest):
    return decision_store.record_decision(request.dict())


@app.get("/api/v1/pricing-decisions")
def list_pricing_decisions(limit: int = 25) -> List[dict]:
    return decision_store.list_decisions(limit=limit)


@app.get("/api/v1/model-health")
def model_health():
    return {
        "model_type": "CatBoostRegressor",
        "prediction_target": "vehicle_market_value_log_price",
        "serving_artifacts": ["artifactS/model.pkl", "artifactS/preprocessor.pkl"],
        "status": "loaded_on_first_request",
    }
