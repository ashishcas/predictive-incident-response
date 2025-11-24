# Predictive Incident Response Agent

This project demonstrates a multi-agent system built with the Google Agent Development Kit (ADK) for ingesting and analyzing log files. The system is designed to process logs, extract key features, and identify potential anomalies, serving as a foundational step in a predictive incident response pipeline.

## 🚀 Features

- **Log Ingestion**: Reads log data from a specified JSON file (`synthetic_nodejs_logs.json`).
- **Feature Extraction**: Calculates metrics from the logs, such as total log count, error counts, 5xx status code counts, and error rates.
- **Sequential Orchestration**: Uses a `SequentialAgent` to automatically run the ingestion and feature extraction steps in order.
- **Artifact Generation**: Saves the processed logs (`ingested_logs_output.json`) and extracted features (`features.json`) into the `session_artifacts/` directory.

## 🛠️ Setup

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites

- Python 3.9+
- A Google API key with the Gemini API enabled. You can get one from Google AI Studio.

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd predictive-incident-response
```

### 3. Set up a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
# Create the virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# .\venv\Scripts\activate
```

### 4. Install Dependencies

This project relies on the Google ADK and other libraries.

```bash
pip install google-adk python-dotenv
```

### 5. Configure Environment Variables

Create a file named `.env` in the root of the project directory and add your Google API key:

```
GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

## ▶️ How to Run

Execute the main agent script from your terminal:

```bash
python agent.py
```

You will see output in the console as the agent ingests the logs and then extracts the features, saving the results in the `session_artifacts` directory.
