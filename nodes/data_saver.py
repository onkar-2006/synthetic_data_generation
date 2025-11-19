from typing import Dict, Any
from core.graph_state import GenerationState
import pandas as pd


def data_saver(state: GenerationState) -> Dict[str, Any]:
    """
    LangGraph node: Saves the final synthetic data and reports.
    In a FastAPI app, this updates the MySQL status and stores the file.
    """

    synthetic_df = state['synthetic_data']
    project_id = state['project_id']

    # --- MVP Logic: Save file locally ---
    output_path = f"synthetic_data_{project_id}.csv"
    synthetic_df.to_csv(output_path, index=False)

    final_status = state['status']
    log = f"Persistence Agent: Final status '{final_status}'. Data saved to {output_path}."

    # Final return for the state before END
    return {'log_messages': state.get('log_messages', []) + [log]}