import unittest
from tempfile import TemporaryDirectory

from src.pricing.decision_store import DecisionStore
from src.pricing.recommendation_service import PricingRecommendationService


class PricingRecommendationTests(unittest.TestCase):
    def setUp(self):
        self.vehicle = {
            "brand": "Ford",
            "model": "Utility Police Interceptor Base",
            "model_year": 2013,
            "milage": "51,000 mi.",
            "fuel_type": "E85 Flex Fuel",
            "engine": "300.0HP 3.7L V6 Cylinder Engine Flex Fuel Capability",
            "transmission": "6-Speed A/T",
            "ext_col": "Black",
            "int_col": "Black",
            "accident": "At least 1 accident or damage reported",
            "clean_title": "Yes",
        }

    def test_recommendation_contains_operational_decision_fields(self):
        service = PricingRecommendationService()
        recommendation = service.recommend_offer(
            self.vehicle,
            {"reconditioning_cost": 1200, "target_margin_pct": 0.12},
        )

        self.assertGreater(recommendation["market_value"], 0)
        self.assertGreater(recommendation["recommended_offer"], 0)
        self.assertIn(recommendation["risk_level"], ["Low", "Medium", "High"])
        self.assertIn("recommended_action", recommendation)
        self.assertIn("explanation", recommendation)

    def test_decision_store_records_override(self):
        with TemporaryDirectory() as tmp_dir:
            store = DecisionStore(f"{tmp_dir}/pricing_decisions.db")
            result = store.record_decision(
                {
                    "quote_id": "Q-TEST123",
                    "user_id": "pricing_user_01",
                    "decision": "override",
                    "recommended_offer": 12900,
                    "final_offer": 13200,
                    "reason": "Manager approved higher offer.",
                }
            )

            self.assertEqual(result["status"], "recorded")
            self.assertEqual(len(store.list_decisions(limit=5)), 1)


if __name__ == "__main__":
    unittest.main()
