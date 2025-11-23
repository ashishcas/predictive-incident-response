import asyncio
import json
import uuid
import os
import dotenv
from typing import Any, Dict

# ADK Imports
from google.adk.agents import Agent, SequentialAgent
from google.adk.models import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types

# Load environment variables
dotenv.load_dotenv()

ARTIFACTS_DIR = "session_artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

print("✅ ADK components imported successfully.")

# --- Configuration ---
MODEL_NAME = "gemini-2.5-flash-lite"
APP_NAME = "log_ingestion_app"
USER_ID = "test_user"
SESSION_NAME = "ingestion_session"
LOG_FILE_PATH = '/Users/ashish/Documents/genai/ai_agents/google-adk/capestoneProject/synthetic_nodejs_logs.json'

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

def ingest_logs(log_file: str) -> str:
    """
    Ingests logs from a source file, saves them to session storage, 
    and returns the storage path.
    """
    # Handle the input path
    base_path = os.getcwd() # Gets the current folder where python is running
    full_artifact_dir = os.path.join(base_path, ARTIFACTS_DIR)
    source_path = LOG_FILE_PATH if "synthetic_nodejs_logs.json" in log_file else log_file
    
    try:
        with open(source_path, 'r') as f:
            raw_data = json.load(f)

        if isinstance(raw_data, list):
            logs = raw_data
        else:
            logs = [hit['_source'] for hit in raw_data.get('hits', {}).get('hits', [])]

        output_filename = f"ingested_logs_output.json"
        output_path = os.path.join(full_artifact_dir, output_filename)
        
        with open(output_path, 'w') as f_out:
            json.dump(logs, f_out, indent=2)

        count = len(logs)
        print(f"DEBUG: 📂 File saved to -> {output_path}")
        return f"✅ Success. Ingested {count} logs. Data saved to artifact: '{output_path}'"

    except FileNotFoundError:
        return f"Error: Source file not found at {source_path}"
    except Exception as e:
        return f"Error processing logs: {str(e)}"

# --- Helper Functions ---

async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
    session_service: InMemorySessionService = None,
):
    print(f"\n ### Session: {session_name}")

    # Get app name from the Runner
    app_name = runner_instance.app_name

    # Attempt to create a new session or retrieve an existing one
    try:
        session = session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except Exception as e:
        session = session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    if user_queries:
        if isinstance(user_queries, str):
            user_queries = [user_queries]

        for query in user_queries:
            print(f"\n👤 User > {query}")
            query_content = types.Content(role="user", parts=[types.Part(text=query)])

            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query_content
            ):
                # Handle the content parts safely
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        # Case 1: It's a Function Call (The Agent is acting)
                        if part.function_call:
                            print(f"   🛠️  [Agent Tool Call] Action: {part.function_call.name}")
                        
                        # Case 2: It's a Text Response (The Agent is speaking)
                        elif part.text:
                             # Filter out empty/system strings
                            if part.text != "None" and part.text.strip():
                                print(f"🤖 {MODEL_NAME} > {part.text}")
    else:
        print("No queries!")


def create_log_ingestor_agent(model: Gemini) -> Agent:
    """An agent that ingests and processes log data from various sources."""
    return Agent(
        model=model,
        name="log_ingestor_agent",
        description="Ingests and processes log data from various sources.",
        tools=[FunctionTool(ingest_logs)],
    )

async def get_orchestrator_agent() -> SequentialAgent:
    """An agent that orchestrates various sub-agents to perform complex tasks."""
    
    # Initialize Model
    orchestrator_model = Gemini(
        model=MODEL_NAME,
        retry_options=retry_config,
    )

    # Initialize Sub-Agent
    log_ingestor = create_log_ingestor_agent(model=orchestrator_model)

    # Return the Composite Agent
    return SequentialAgent(
        name="capstone_orchestrator",
        sub_agents=[log_ingestor]
    )

# --- Main Execution ---

async def main():
    print("🧪 Initializing Log Ingestor System...")

    session_service = InMemorySessionService()    
    
    root_agent = await get_orchestrator_agent()

    runner = Runner(
        app_name=APP_NAME, 
        agent=root_agent, 
        session_service=session_service
    )
    
    user_query = "Please ingest the logs from 'synthetic_nodejs_logs.json'"
    
    await run_session(
        runner,
        user_queries=[user_query],
        session_name=SESSION_NAME,
        session_service=session_service,
    )
    
    print("\n✅ Test completed.")

if __name__ == "__main__":
    asyncio.run(main())