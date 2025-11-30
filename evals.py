import unittest
import os
import json
import shutil
import asyncio
from agent import (
    ingest_logs,
    extract_features,
    predict_failure_risk,
    perform_root_cause_analysis,
    generate_remediation_plan,
    PIPELINE_STATE,
    ARTIFACTS_DIR,
    LOG_FILE_PATH
)

class TestAgentTools(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure artifacts directory exists and is clean for testing
        if os.path.exists(ARTIFACTS_DIR):
            shutil.rmtree(ARTIFACTS_DIR)
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        
        # Create a dummy log file for testing if it doesn't exist
        if not os.path.exists(LOG_FILE_PATH):
            dummy_logs = [
                {"log": {"level": "INFO"}, "http": {"response": {"status_code": 200}}},
                {"log": {"level": "ERROR"}, "http": {"response": {"status_code": 500}}},
                {"log": {"level": "WARN"}, "http": {"response": {"status_code": 404}}},
            ]
            with open(LOG_FILE_PATH, 'w') as f:
                json.dump(dummy_logs, f)

    def test_01_ingest_logs(self):
        print("\nTesting ingest_logs...")
        result = ingest_logs(LOG_FILE_PATH)
        print(result)
        self.assertIn("Success", result)
        self.assertIsNotNone(PIPELINE_STATE["latest_log_artifact"])
        self.assertTrue(os.path.exists(PIPELINE_STATE["latest_log_artifact"]))

    def test_02_extract_features(self):
        print("\nTesting extract_features...")
        # Ensure state is set (dependent on previous test or manual setup)
        if not PIPELINE_STATE["latest_log_artifact"]:
             ingest_logs(LOG_FILE_PATH)
             
        result = extract_features()
        print(result)
        self.assertIn("Feature extraction complete", result)
        self.assertIsNotNone(PIPELINE_STATE["latest_feature_artifact"])
        self.assertTrue(os.path.exists(PIPELINE_STATE["latest_feature_artifact"]))
        
        # Verify content
        with open(PIPELINE_STATE["latest_feature_artifact"], 'r') as f:
            features = json.load(f)
            self.assertIn("error_rate_percent", features)

    def test_03_predict_failure_risk(self):
        print("\nTesting predict_failure_risk...")
        if not PIPELINE_STATE["latest_feature_artifact"]:
            ingest_logs(LOG_FILE_PATH)
            extract_features()

        result = predict_failure_risk()
        print(result)
        # Result is a JSON string
        data = json.loads(result)
        self.assertIn("failure_probability", data)
        self.assertIsNotNone(PIPELINE_STATE["latest_prediction_artifact"])

    def test_04_perform_root_cause_analysis(self):
        print("\nTesting perform_root_cause_analysis...")
        # Setup state
        if not PIPELINE_STATE["latest_prediction_artifact"]:
             ingest_logs(LOG_FILE_PATH)
             extract_features()
             predict_failure_risk()

        result = perform_root_cause_analysis()
        print(result)
        data = json.loads(result)
        self.assertIn("current_incident_evidence", data)
        self.assertIn("similar_past_incidents", data)

    def test_05_generate_remediation_plan(self):
        print("\nTesting generate_remediation_plan...")
        # Mock RCA output
        rca_output = "The root cause is likely Database Connection Pool exhaustion."
        
        result = generate_remediation_plan(rca_output)
        print(result)
        data = json.loads(result)
        self.assertIn("action_plan", data)
        self.assertEqual(data["root_cause_category"], "Database Connection Pool")

if __name__ == '__main__':
    unittest.main()
