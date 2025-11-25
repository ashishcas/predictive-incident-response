# 🤖 Multi-Agent Observability & Predictive Incident Response System

![System Status](https://img.shields.io/badge/System-Operational-green) ![Agents](https://img.shields.io/badge/Agents-6_Active-blue) ![Powered By](https://img.shields.io/badge/Powered_By-Google_ADK_%26_Gemini-orange)

An intelligent, multi-agent system designed to predict, analyze, and remediate system failures autonomously. It leverages **Google ADK** and **Gemini** to ingest logs, detect anomalies, perform Root Cause Analysis (RCA), and suggest infrastructure fixes.

---

## 🏗️ Architecture

The system operates as a sequential pipeline of specialized agents:

1.  **Log Ingestor:** Fetches and normalizes raw logs from disparate sources.
2.  **Preprocessor:** Engineers features (Error Rates, Latency trends) from raw data.
3.  **Anomaly Detector:** Uses LLM reasoning (SRE Persona) to identify critical failures.
4.  **Predictor:** Forecasts future failure probabilities using weighted risk modeling.
5.  **RCA Agent:** Correlates current evidence with past incidents to find the root cause.
6.  **Remediation Agent:** Generates actionable Kubernetes/Infrastructure commands to fix the issue.

---

## 🚀 Execution Demo

Below is an actual execution log from a simulated "Database Connection Timeout" incident.

### **1. Initialization & Ingestion**
**User Query:** *"Please ingest the logs from 'synthetic_nodejs_logs.json'..."*
- **Action:** `ingest_logs`
- **Result:** Logs successfully parsed.
- **Artifact:** `session_artifacts/ingested_logs_output.json`

### **2. Feature Engineering**
The Preprocessor analyzed the raw logs to calculate key stability metrics.
- **Metrics:**
  - `total_logs`: 100
  - `error_count`: 13
  - `error_rate_percent`: **13.0%** (Critical High)
  - `is_anomaly_suspect`: `True`

### **3. Anomaly Detection (SRE Persona)**
The SRE Agent analyzed the metrics and flagged a critical issue.
> **🚨 ANOMALY DETECTED:** "The error rate is 13.0%, which is above the 3% threshold. Additionally, there are 13 instances of HTTP 5xx errors, indicating critical failures."

### **4. Root Cause Analysis (RCA)**
The RCA Agent correlated the logs with past incident history.
- **Hypothesis:** Database Connection Timeout in `payment-service`.
- **Evidence:** `message: "Database connection timeout [req_id=9360]"`
- **Confidence:** **98%**
- **Match:** Matched past incident `INC-2023-001` (DB Pool Exhaustion).

### **5. Remediation Plan**
The DevOps Agent generated the following infrastructure runbook:

| Action | Command | Risk Level | Status |
| :--- | :--- | :--- | :--- |
| **Scale Up DB** | `kubectl scale statefulset payment-db --replicas=5` | 🔴 **HIGH** | `PENDING_APPROVAL` |
| **Restart Service** | `kubectl rollout restart deployment/payment-service` | 🟠 **MEDIUM** | `PENDING_APPROVAL` |

---

## 🛠️ Setup & Installation

### Prerequisites
* Python 3.10+
* Google Cloud Project with Vertex AI enabled
* Google ADK installed

### Installation
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ashishcas/predictive-incident-response.git](https://github.com/ashishcas/predictive-incident-response.git)
    ```
