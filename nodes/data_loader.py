import pandas as pd
from typing import Dict, Any
from core.graph_state import GenerationState


def data_loader(state: GenerationState) -> Dict[str, Any]:
    """LangGraph node: Loads the input file (CSV) into a Pandas DataFrame."""

    file_path = state['input_file_path']
    project_id = state['project_id']

    try:
        df = pd.read_csv(file_path)
        log = f"Project {project_id}: Data loaded successfully. Shape: {df.shape}"

        return {
            'original_data': df,
            'status': 'Data Loaded',
            'log_messages': state.get('log_messages', []) + [log]
        }
    except Exception as e:
        error_msg = f"Error loading data: {e}"
        return {
            'status': 'Error',
            'error_message': error_msg,
            'log_messages': state.get('log_messages', []) + [error_msg]
        }