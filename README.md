## Agentic Synthetic Data Pipeline

This project implements a web-based interface to an Agentic Synthetic Data Generation Pipeline. The pipeline uses LangGraph and Large Language Models (LLMs) to automatically infer data schema from an input CSV file, generate new synthetic data that mirrors the original's distribution, and perform a quality check before approving the output.

The application is split into two main parts: a FastAPI backend that runs the pipeline, and a pure HTML/JavaScript frontend for user interaction and real-time status monitoring.

🚀 Features

File Upload: Users can upload their own .csv files via the web interface.

Schema Inference: The pipeline automatically analyzes the input data to determine column types and relationships.

Synthetic Data Generation: Generates new data based on the inferred schema and distribution.

Quality Vetting: Calculates an Overall Quality Score and approves the data for use.

Real-time Monitoring: The frontend polls the server to display the current pipeline status, quality score, and data preview.

CORS Enabled: Configured for local development compatibility.

🏛️ Architecture Overview

The system consists of three main components:

Frontend (index.html): A single HTML file containing all presentation (Tailwind CSS) and logic (JavaScript). It handles file selection, triggering the pipeline via API calls, and displaying results in real-time.

API Backend (api_server.py): Built with FastAPI, this server exposes endpoints for:

/upload-file (POST): Handles incoming CSV files.

/run-pipeline (POST): Triggers the LangGraph execution.

/status (GET): Provides real-time status updates and quality score.

/data (GET): Retrieves the final synthetic data for preview.

Pipeline Core (main_graph.py, core/...): The core intelligence running on the backend, responsible for the multi-step, agentic data generation process using LangGraph.

🛠️ Prerequisites

Before starting, ensure you have the following installed:

Python 3.9+

pip (Python package installer)

LangChain/LangGraph Dependencies: The backend relies on several Python libraries.

You can install the necessary Python packages using pip. Run this command in your project's virtual environment:

pip install fastapi uvicorn pandas python-multipart requests
# You also need the packages for the LangGraph/LLM core (LangChain, LangGraph, LLM integration, etc.)
# If you are using a specific LLM integration, include those packages here (e.g., 'llama-cpp-python', 'huggingface-hub')


### ⚙️ Setup and Installation

Project Structure: Ensure your files are structured correctly:

/your_project_directory
├── api_server.py           # The FastAPI backend (MUST be run via uvicorn)
├── index.html              # The frontend (open in browser)
├── main_graph.py           # (Assumed) LangGraph definition
├── core/                   # (Assumed) Pipeline modules and state
└── uploads/                # (Automatically created) Directory for uploaded files


Input Data: For initial testing, ensure you have a dummy input file ready. The server is configured to use dummy_input.csv if no file is uploaded.

▶️ Running the Application

You must run the backend server first, and then open the frontend in your browser.

1. Start the API Server

Use Uvicorn to run the FastAPI application. Do not run python api_server.py directly.

Open your terminal or command prompt.

Navigate to your project directory.

Run the server:

uvicorn api_server:app --reload


You should see a message confirming the server is running, typically at http://127.0.0.1:8000. Keep this terminal window open.

2. Access the Frontend

Locate the index.html file in your project directory.

Open it directly in your web browser (e.g., Chrome, Firefox).

3. Usage

Upload File: Use the Choose File button to select a .csv file from your local machine. The filename will be displayed next to "Input File."

Run Pipeline: Click the Run Synthetic Data Pipeline button.

Monitor: The Status, Progress Message, and Quality Score will update every 2 seconds as the pipeline progresses on the server.

View Results: Once the status shows Quality Approved, the synthetic data will be loaded and displayed in the Data Preview section.

🛑 Troubleshooting

CORS Policy Blocked Error

If you see errors like Access to fetch at '...' has been blocked by CORS policy, it means your server is not configured to accept requests from your local HTML file.

Fix: This has been resolved in the provided api_server.py by adding the CORSMiddleware. Ensure your running server uses the latest code from the last update. If the error persists, verify that the uvicorn process was restarted after the code change.
