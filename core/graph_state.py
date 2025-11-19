from typing import TypedDict, List, Dict, Any, Union
from pandas import DataFrame

# Define the Agent's shared memory (State)
class GenerationState(TypedDict):

    # 1. Input/Context
    project_id: str
    input_file_path: str
    original_data: DataFrame

    # 2. Schema and Constraints
    inferred_schema: Dict[str, Any]
    user_constraints: List[str]

    # 3. Output Data & Fidelity
    synthetic_data: DataFrame
    quality_report: Dict[str, Union[str, float]]

    # 4. Status & Logging
    status: str
    log_messages: List[str]
    error_message: str | None

