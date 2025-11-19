import json
import os
# FIX: Use the new langchain_huggingface package to support newer huggingface_hub versions
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from typing import Dict, Any
from core.graph_state import GenerationState
from core.tools import analyze_dataframe_stats

# --- Hugging Face LLM Setup ---
HF_REPO_ID = "meta-llama/Llama-2-7b-chat-hf"

try:
    # The environment variable HUGGINGFACEHUB_API_TOKEN must be set.
    hf_llm = HuggingFaceEndpoint(
        repo_id=HF_REPO_ID,
        temperature=0.01,
        max_new_tokens=2048,
        task="text-generation"  # Explicitly define task
    )
    print(f"HuggingFace Endpoint initialized with model: {HF_REPO_ID}")
except Exception as e:
    print(f"Warning: Failed to initialize HuggingFace Endpoint. {e}")
    hf_llm = None


def schema_inference(state: GenerationState) -> Dict[str, Any]:
    """LangGraph node: Uses Hugging Face LLM to infer the schema based on Pandas stats."""

    # 1. Check if LLM is ready
    if hf_llm is None:
        return {
            'status': 'Error',
            'error_message': "Schema Agent Error: LLM not initialized (Check API Token)."
        }

    df = state['original_data']
    log_messages = state.get('log_messages', [])

    # 2. Get the statistical summary from the Tool
    try:
        stats_output = analyze_dataframe_stats.invoke({'df': df})
    except Exception as e:
        return {'status': 'Error', 'error_message': f"Pandas Tool Error: {e}"}

    # 3. Strong Prompt Engineering for JSON Output
    system_prompt = (
        "You are the Schema Inference Agent. Your task is to analyze the provided "
        "data statistics and generate a formal JSON schema for synthetic data generation. "
        "The schema must define column names, their final data types (e.g., 'int', 'float', 'category', 'datetime'), "
        "and any statistical limits (min/max/top_values). "
        "The entire output MUST be a single JSON object. DO NOT include any explanation, markdown fencing, or extra text."
    )

    llm_prompt = f"{system_prompt}\n\nDATA STATISTICS:\n{stats_output}\n\nOutput the complete JSON schema:"

    try:
        # 4. Call the Hugging Face endpoint
        raw_output = hf_llm.invoke(llm_prompt)

        # 5. Cleanup and Parse the JSON Output
        cleaned_output = raw_output.strip()
        # Remove markdown code blocks if present
        if cleaned_output.startswith("```"):
            cleaned_output = cleaned_output.strip("`").replace("json", "").strip()

        # Attempt to parse the cleaned JSON
        inferred_schema = json.loads(cleaned_output)

        # Handle wrapped responses if necessary
        if 'columns' in inferred_schema:
            schema_data = inferred_schema['columns']
        else:
            schema_data = inferred_schema

        log = f"Schema Agent: Schema inferred successfully using {HF_REPO_ID}."
        log_messages.append(log)

        return {
            'inferred_schema': schema_data,
            'status': 'Schema Inferred',
            'log_messages': log_messages
        }

    except json.JSONDecodeError as jde:
        # Fallback: If JSON parsing fails, we can log it but might allow the pipeline
        # to proceed if we trust the SDV auto-detection in the next step.
        # For MVP, we mark it as an error.
        error_msg = f"Schema Agent Error: LLM output was not valid JSON. Output snippet: {cleaned_output[:50]}..."
        log_messages.append(error_msg)
        return {'status': 'Error', 'error_message': error_msg, 'log_messages': log_messages}

    except Exception as e:
        error_msg = f"Schema Agent Error: Failed to invoke Hugging Face Endpoint. {e}"
        log_messages.append(error_msg)
        return {'status': 'Error', 'error_message': error_msg, 'log_messages': log_messages}