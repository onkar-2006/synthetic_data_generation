import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdmetrics.reports.single_table import QualityReport
from typing import Dict, Any
from core.graph_state import GenerationState


def quality_check(state: GenerationState) -> Dict[str, Any]:
    """
    LangGraph node: Validates the synthetic data using structural checks and SDMetrics.
    """
    # --- 🛡️ SAFETY CHECK: Skip validation if previous steps failed ---
    if state.get('status') == 'Error':
        return {
            'status': 'Error',
            'log_messages': state.get('log_messages', []) + ["Validation skipped due to previous error."]
        }

    original_df = state['original_data']
    synthetic_df = state['synthetic_data']
    log_messages = state.get('log_messages', [])

    if synthetic_df is None:
        error_msg = "Critical Error: Synthetic data is missing (None) despite 'Data Generated' status."
        return {
            'status': 'Error',
            'error_message': error_msg,
            'log_messages': log_messages + [error_msg]
        }

    validation_issues = []

    # --- 1. Structural Validation ---
    if synthetic_df.shape[1] != original_df.shape[1]:
        validation_issues.append("ERROR: Column count mismatch.")

    if validation_issues:
        error_msg = f"Structural Validation Failed. {len(validation_issues)} issues found."
        log_messages.append(error_msg)
        return {
            'status': 'Validation Failure',
            'error_message': error_msg + " | Details: " + " ".join(validation_issues),
            'log_messages': log_messages
        }

    # --- 2. Statistical Validation (Fidelity Check) ---
    try:
        # 1. Instantiate metadata object first (Required for SDV 1.0+)
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=original_df)

        report = QualityReport()

        # 2. Convert SingleTableMetadata object to a dictionary for SDMetrics
        report.generate(original_df, synthetic_df, metadata.to_dict())

        overall_score = report.get_score()

        log = f"Statistical Validation Complete. Overall Score: {overall_score:.2f}"
        log_messages.append(log)

        # 3. FIX: Use 'get_properties()' instead of 'get_details()'
        # This returns the summary dataframe (Column Shapes, Trends scores)
        properties_json = report.get_properties().to_json()

        return {
            'quality_report': {
                'Overall Score': overall_score,
                'Details': properties_json
            },
            'status': 'Quality Approved',
            'log_messages': log_messages
        }
    except Exception as e:
        error_msg = f"Statistical Validation Error: SDMetrics failed. {e}"
        return {'status': 'Error', 'error_message': error_msg}