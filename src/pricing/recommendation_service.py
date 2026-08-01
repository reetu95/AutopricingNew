import math
import re
import uuid
from datetime import datetime, timezone

import numpy as np

from src.pipline.predict_pipeline import CustomData, PredictPipeline


CURRENT_YEAR = 2026


def parse_money(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[^0-9.]", "", str(value))
    return float(cleaned) if cleaned else None


def parse_mileage(value):
    parsed = parse_money(value)
    return int(parsed) if parsed is not None else 0


def round_to_nearest_100(value):
    return int(round(value / 100.0) * 100)


class PricingRecommendationService:
    def __init__(self):
        self.predict_pipeline = PredictPipeline()

    def predict_market_value(self, vehicle):
        custom_data = CustomData(
            brand=vehicle["brand"],
            model=vehicle["model"],
            model_year=int(vehicle["model_year"]),
            milage=vehicle["milage"],
            fuel_type=vehicle["fuel_type"],
            engine=vehicle["engine"],
            transmission=vehicle["transmission"],
            ext_col=vehicle["ext_col"],
            int_col=vehicle["int_col"],
            accident=vehicle["accident"],
            clean_title=vehicle["clean_title"],
        )
        prediction_frame = custom_data.get_data_as_data_frame()
        log_prediction = self.predict_pipeline.predict(prediction_frame)
        return float(np.exp(log_prediction[0]))

    def recommend_offer(self, vehicle, pricing_context):
        market_value = self.predict_market_value(vehicle)
        mileage = parse_mileage(vehicle["milage"])
        vehicle_age = max(CURRENT_YEAR - int(vehicle["model_year"]), 0)
        reconditioning_cost = pricing_context.get("reconditioning_cost", 1200.0)
        target_margin_pct = pricing_context.get("target_margin_pct", 0.12)

        risk_score, risk_factors = self._score_risk(vehicle, mileage, vehicle_age)
        risk_adjustment = market_value * risk_score * 0.18
        target_margin_amount = max(market_value * target_margin_pct, 750.0)

        raw_offer = market_value - reconditioning_cost - target_margin_amount - risk_adjustment
        floor_offer = market_value * 0.52
        ceiling_offer = market_value * 0.92
        recommended_offer = min(max(raw_offer, floor_offer), ceiling_offer)
        recommended_offer = round_to_nearest_100(recommended_offer)

        expected_margin = market_value - recommended_offer - reconditioning_cost
        expected_margin_pct = expected_margin / market_value if market_value else 0
        risk_level = self._risk_level(risk_score)
        approval_required = (
            risk_level == "High"
            or expected_margin_pct < 0.08
            or recommended_offer >= 35000
        )

        action = self._recommended_action(approval_required, risk_level, expected_margin_pct)
        candidate_offers = self._candidate_offers(market_value, reconditioning_cost)

        return {
            "quote_id": f"Q-{uuid.uuid4().hex[:10].upper()}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "market_value": round(market_value, 2),
            "recommended_offer": recommended_offer,
            "expected_margin": round(expected_margin, 2),
            "expected_margin_pct": round(expected_margin_pct, 4),
            "risk_score": round(risk_score, 3),
            "risk_level": risk_level,
            "approval_required": approval_required,
            "recommended_action": action,
            "risk_factors": risk_factors,
            "candidate_offers": candidate_offers,
            "explanation": self._explain(vehicle, market_value, risk_factors, action),
        }

    def _score_risk(self, vehicle, mileage, vehicle_age):
        score = 0.08
        factors = []

        if vehicle_age >= 12:
            score += 0.14
            factors.append("Vehicle age is above 12 years.")
        elif vehicle_age >= 8:
            score += 0.08
            factors.append("Vehicle age is above 8 years.")

        if mileage >= 150000:
            score += 0.16
            factors.append("Mileage is above 150,000 miles.")
        elif mileage >= 100000:
            score += 0.10
            factors.append("Mileage is above 100,000 miles.")
        elif mileage >= 70000:
            score += 0.05
            factors.append("Mileage is above 70,000 miles.")

        accident = str(vehicle["accident"]).lower()
        if "accident" in accident or "damage" in accident:
            score += 0.12
            factors.append("Accident or damage history increases pricing risk.")

        clean_title = str(vehicle["clean_title"]).lower()
        if clean_title != "yes":
            score += 0.16
            factors.append("Clean title is missing or not confirmed.")

        fuel_type = str(vehicle["fuel_type"]).lower()
        if "not supported" in fuel_type or fuel_type in {"nan", "none", ""}:
            score += 0.07
            factors.append("Fuel type is missing or unsupported.")

        return min(score, 0.65), factors or ["No major risk flags found."]

    def _risk_level(self, risk_score):
        if risk_score >= 0.32:
            return "High"
        if risk_score >= 0.20:
            return "Medium"
        return "Low"

    def _recommended_action(self, approval_required, risk_level, expected_margin_pct):
        if approval_required:
            return "Send to manager review before making offer."
        if risk_level == "Medium":
            return "Make offer with negotiation buffer."
        if expected_margin_pct >= 0.16:
            return "Make offer confidently."
        return "Make offer and monitor margin."

    def _candidate_offers(self, market_value, reconditioning_cost):
        offers = []
        for discount_pct in [0.08, 0.12, 0.16, 0.20]:
            offer = round_to_nearest_100(market_value * (1 - discount_pct) - reconditioning_cost)
            margin = market_value - offer - reconditioning_cost
            offers.append(
                {
                    "discount_pct": discount_pct,
                    "offer": offer,
                    "expected_margin": round(margin, 2),
                    "expected_margin_pct": round(margin / market_value, 4),
                }
            )
        return offers

    def _explain(self, vehicle, market_value, risk_factors, action):
        primary_factors = ", ".join(
            [
                vehicle["brand"],
                vehicle["model"],
                str(vehicle["model_year"]),
                str(vehicle["milage"]),
            ]
        )
        return [
            f"Predicted market value is based on vehicle attributes: {primary_factors}.",
            f"Risk adjustments considered: {' '.join(risk_factors)}",
            f"Recommended action: {action}",
            f"Estimated market value used by pricing engine: ${market_value:,.0f}.",
        ]
