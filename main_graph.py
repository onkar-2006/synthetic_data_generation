import pandas as pd
from langgraph.graph import StateGraph, END
from typing import Dict, Any

# Import the shared state
from core.graph_state import GenerationState

# Import the nodes
from nodes.data_loader import data_loader
from nodes.schema_inference import schema_inference
from nodes.data_generation import data_generation
from nodes.quality_check import quality_check
from nodes.data_saver import data_saver


# --- Build the LangGraph ---
def build_generation_graph():
    builder = StateGraph(GenerationState)

    # 1. Add the nodes
    builder.add_node("data_loader", data_loader)
    builder.add_node("schema_inference", schema_inference)
    builder.add_node("data_generation", data_generation)
    builder.add_node("quality_check", quality_check)
    builder.add_node("data_saver", data_saver)

    # 2. Define the Edges (Sequential Flow)
    builder.set_entry_point("data_loader")
    builder.add_edge("data_loader", "schema_inference")
    builder.add_edge("schema_inference", "data_generation")
    builder.add_edge("data_generation", "quality_check")

    # 3. Define the Conditional Edge for Error Handling
    def is_approved(state: GenerationState):
        """Checks if the pipeline status is approved or failed."""
        # If any previous step set status to Error, we stop.
        if state['status'] in ['Error', 'Validation Failure']:
            return "end_with_error"
        return "approved"

    builder.add_conditional_edges(
        "quality_check",
        is_approved,
        {
            "approved": "data_saver",
            "end_with_error": END  # Immediately stop the graph on failure
        }
    )

    # 4. Final Edges
    builder.add_edge("data_saver", END)

    return builder.compile()


# Example Usage (Used for testing/running the graph)
if __name__ == "__main__":
    # Create a dummy CSV file for demonstration
    dummy_file = "dummy_input.csv"
    dummy_data = {
        'ID': range(100),
        'Age': pd.Series([25, 30, 45, 60] * 25),
        'Salary': pd.Series([50000, 75000, 120000, 45000] * 25),
        'City': pd.Series(['NY', 'LA', 'NY', 'SF'] * 25)
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_df.to_csv(dummy_file, index=False)
    print(f"Created dummy input file: {dummy_file}")

    app_graph = build_generation_graph()

    initial_state = GenerationState(
        project_id="P_001",
        input_file_path=dummy_file,
        original_data=None,  # Loaded by data_loader
        inferred_schema={},
        user_constraints=[],
        synthetic_data=None,
        quality_report={},
        status='Initialized',
        log_messages=[],
        error_message=None
    )

    print("--- Starting Agentic Synthetic Data Pipeline ---")
    try:
        final_state = app_graph.invoke(initial_state)

        print("\n--- Final State Summary ---")
        print(f"Final Status: {final_state['status']}")

        # --- 🔍 ERROR DEBUGGING: Print specific details if it failed ---
        if final_state['status'] == 'Error':
            print(f"\n❌ ERROR DETAILS: {final_state.get('error_message', 'No error message found.')}")
            print("\n--- Execution Log History ---")
            for i, log in enumerate(final_state.get('log_messages', []), 1):
                print(f"{i}. {log}")
        else:
            print(f"Overall Quality Score: {final_state.get('quality_report', {}).get('Overall Score', 'N/A')}")

    except Exception as e:
        print(f"\n❌ CRITICAL GRAPH EXECUTION ERROR: {e}")