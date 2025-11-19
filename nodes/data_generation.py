import io
from typing import Dict, Any
from core.graph_state import GenerationState
from core.tools import generate_synthetic_data_tool  # Import the tool
import pandas as pd


def data_generation(state: GenerationState) -> Dict[str, Any]:
    """LangGraph node: Orchestrates the synthetic data generation using SDV Tool."""

    df = state['original_data']
    original_data_csv = df.to_csv(index=False)

    # Execute the synthesis tool
    result_csv_str = generate_synthetic_data_tool.invoke({
        'original_data_csv': original_data_csv,
        'num_rows': 5000
    })

    if result_csv_str.startswith("SYNTHESIS_ERROR"):
        return {'status': 'Error', 'error_message': result_csv_str}

    synthetic_df = pd.read_csv(io.StringIO(result_csv_str))

    log = f"Synthesis Agent: Generated {synthetic_df.shape[0]} synthetic rows."
    return {
        'synthetic_data': synthetic_df,
        'status': 'Data Generated',
        'log_messages': state.get('log_messages', []) + [log]
    }