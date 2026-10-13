import json
import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import gtm_agent


PROSPECT_ID = "LEAD-PRIVACY-1"
SENSITIVE_FIELDS = {
    "tax_id": "123-45-6789",
    "date_of_birth": "1980-01-02",
    "card_on_file": "4111111111111111",
    "credit_check_ref": "EXPN-12345",
}
PROSPECT_RECORD = {
    "prospect_id": PROSPECT_ID,
    "name": "Alex Example",
    "email": "alex@example.com",
    "disqualified": False,
    "enrichment_source": "Lusha",
    "annual_revenue": 60000000,
    "billing_qualification": SENSITIVE_FIELDS,
    "engagement_history": ["Attended a webinar"],
    "account_details": {"industry": "Technology"},
    "tech_stack": ["Salesforce"],
    "other_internal_field": "not for the agent",
}
OFFERING = {
    "required_tech_stack": ["Salesforce"],
    "min_annual_revenue": 10000000,
    "description": "GTM platform",
}


def assert_no_sensitive_keys(test_case, value):
    if isinstance(value, dict):
        for key, nested_value in value.items():
            test_case.assertNotIn(key, SENSITIVE_FIELDS)
            assert_no_sensitive_keys(test_case, nested_value)
    elif isinstance(value, (list, tuple)):
        for nested_value in value:
            assert_no_sensitive_keys(test_case, nested_value)


class ProspectPrivacyTests(unittest.TestCase):
    def test_get_prospect_returns_only_contact_allowlist(self):
        with patch.object(gtm_agent.data_service, "get_prospect_record", return_value=PROSPECT_RECORD):
            result = gtm_agent.get_prospect.invoke({"prospect_id": PROSPECT_ID})

        self.assertEqual(
            result["prospect"],
            {
                "prospect_id": PROSPECT_ID,
                "name": "Alex Example",
                "email": "alex@example.com",
                "disqualified": False,
                "enrichment_source": "Lusha",
            },
        )
        assert_no_sensitive_keys(self, result)

    def test_build_profile_and_cache_hit_exclude_sensitive_fields(self):
        with (
            patch.object(gtm_agent.data_service, "get_profile_from_db", return_value={"prospect_profile": None}),
            patch.object(gtm_agent.data_service, "get_prospect_record", return_value=PROSPECT_RECORD),
            patch.object(gtm_agent.data_service, "fetch_engagement_history", return_value=PROSPECT_RECORD["engagement_history"]),
            patch.object(gtm_agent.data_service, "fetch_account_details", return_value=PROSPECT_RECORD["account_details"]),
            patch.object(gtm_agent.data_service, "fetch_tech_stack", return_value=PROSPECT_RECORD["tech_stack"]),
            patch.object(gtm_agent.data_service, "save_profile_to_db") as save_profile,
        ):
            built = gtm_agent.build_prospect_profile.invoke({"prospect_id": PROSPECT_ID})

        expected_profile = {
            "prospect_id": PROSPECT_ID,
            "name": "Alex Example",
            "annual_revenue": 60000000,
            "tech_stack": ["Salesforce"],
            "account_details": {"industry": "Technology"},
            "engagement_history": ["Attended a webinar"],
        }
        self.assertEqual(built["prospect_profile"], expected_profile)
        self.assertEqual(save_profile.call_args.args[1], expected_profile)
        assert_no_sensitive_keys(self, built)

        with (
            patch.object(gtm_agent.data_service, "get_profile_from_db", return_value={"prospect_profile": {
                **expected_profile,
                "billing_qualification": SENSITIVE_FIELDS,
                "email": PROSPECT_RECORD["email"],
            }}),
            patch.object(gtm_agent.data_service, "save_profile_to_db") as save_profile,
        ):
            cached = gtm_agent.build_prospect_profile.invoke({"prospect_id": PROSPECT_ID})

        self.assertEqual(cached["prospect_profile"], expected_profile)
        self.assertEqual(save_profile.call_args.args[1], expected_profile)
        assert_no_sensitive_keys(self, cached)

    def test_score_prospect_sends_only_scoring_allowlist(self):
        profile = {
            **{key: PROSPECT_RECORD[key] for key in (
                "prospect_id", "name", "annual_revenue", "tech_stack",
                "account_details", "engagement_history",
            )},
            "billing_qualification": SENSITIVE_FIELDS,
            "email": PROSPECT_RECORD["email"],
        }
        scoring_result = Mock()
        scoring_result.model_dump.return_value = {"score": 90}

        with (
            patch.object(gtm_agent.data_service, "fetch_tech_stack", return_value=["Salesforce"]),
            patch.object(gtm_agent, "_scoring_llm", Mock(invoke=Mock(return_value=scoring_result))) as scoring_llm,
        ):
            result = gtm_agent.score_prospect.invoke({"prospect_profile": profile, "offering": OFFERING})

        self.assertEqual(result, {"score": 90})
        user_message = scoring_llm.invoke.call_args.args[0][1]["content"]
        sent_profile = json.loads(user_message.split("\n\nProspect profile:\n", 1)[1])
        self.assertEqual(
            sent_profile,
            {key: profile[key] for key in gtm_agent.SCORING_FIELDS},
        )
        assert_no_sensitive_keys(self, scoring_llm.invoke.call_args.args)
        for sensitive_value in SENSITIVE_FIELDS.values():
            self.assertNotIn(sensitive_value, user_message)


if __name__ == "__main__":
    unittest.main()
